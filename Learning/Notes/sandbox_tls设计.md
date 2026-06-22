# Sandbox TLS 证书管理设计文档

> 范围：intelliRouter 中涉及 TLS 证书的全生命周期管理，包括配置、加载、握手、热重载、连接复用与安全响应。
>
> 关联文档：[sandbox设计.md](sandbox设计.md)（Sandbox 模块总体设计）、[identity设计.md](identity设计.md)、[基础框架设计.md](基础框架设计.md)。

---

## 一、功能设计

### 1.1 模块概述

intelliRouter 在与 NaCRE 服务通信、接收沙箱注册回调时**全程使用 mTLS（双向认证）**。证书管理涵盖从配置加载、TLS 握手、连接池复用到运行时热重载的完整链路，是 sandbox 模块可靠性和安全性的基石。

**证书在系统中的位置**：

```
┌────────────────┐                                  ┌────────────────┐
│ intelliRouter  │  ──── mTLS (outbound) ────────►  │   NaCRE 服务   │
│  (NacreClient) │  ◄───── mTLS (inbound) ────────  │                │
└────────────────┘                                  └────────────────┘
        ▲
        │ mTLS (inbound)
        │ 沙箱启动后回调 intelliRouter
        ▼
┌────────────────┐
│  沙箱执行环境   │
│ (NacreListener)│
└────────────────┘
```

**两套证书，分别服务两个方向**：

| 证书 | 配置字段 | 用途 | 验证方向 |
|------|---------|------|---------|
| NaCRE 客户端证书 | `sandbox_proxy.nacre.tls` | 我们调用 NaCRE | NaCRE 验证"我们" |
| 沙箱回调证书 | `sandbox_proxy.listeners.tls` | 沙箱回调我们 | 沙箱验证"我们" |

### 1.2 核心功能

#### 1.2.1 证书加载
- **启动时**：从磁盘 PEM 文件读取证书与私钥，加密私钥经 SCC 解密后载入内存
- **懒加载**：NacreClient 的 `Client` 实例在第一次 HTTPS 请求时才构建，不阻塞进程启动
- **配置驱动**：通过 `TlsConfig` 控制 TLS 开关、CA 证书路径、本端证书/私钥路径

#### 1.2.2 TLS 握手
- **mTLS 双向认证**：两端互验证书，防止中间人和假冒
- **CA 证书池**：客户端使用配置的 CA 验证 NaCRE；服务端使用 CA + `WebPkiClientVerifier` 验证沙箱
- **HTTP/1.1 keep-alive**：握手完成后连接保留，供后续请求复用

#### 1.2.3 连接池复用
- **客户端**：`OnceCell<Client>` 缓存 `hyper_util::Client`，跨请求复用 TLS 连接
- **空闲超时**：`pool_idle_timeout = 90s` 防止 NAT/防火墙静默断开造成"连接已死"问题
- **服务端**：`hyper::Server` 的 keep-alive 天然实现连接复用

#### 1.2.4 运行时热重载
- **零重启**：通过 Unix Domain Socket 接收 `TlsReload` 通知，无需重启进程
- **客户端**：清空 `OnceCell`，下一次请求触发重建（用新证书）
- **服务端**：`DynamicCertResolver` 原地换 `Arc<CertifiedKey>`，ServerConfig 引用不变
- **保证进行中请求不中断**：新客户端的 `OnceCell` 释放时旧 `Client` 引用计数归零才 drop

#### 1.2.5 超时与可靠性
- **请求超时**：所有 `client.request()` 用 `tokio::time::timeout` 包裹，覆盖配置中的 `client_config.timeout`
- **解锁释放**：超时触发后立即释放 `nacre_client` 上的读锁，避免证书重载等写锁被永久阻塞
- **错误降级**：超时、连接拒绝、TLS 错误统一转换为 `anyhow::Error` 透传

### 1.3 设计目标

1. **零停机证书轮换**：从触发重载到生效，0 秒中断、0 个失败请求
2. **私钥安全**：磁盘加密存储 + 内存 `Zeroizing` 清零 + 限定解析生命周期
3. **写锁安全**：业务请求必有上限时间，保证运维通道不卡死
4. **连接复用**：单次 TLS 握手在 keep-alive 窗口内可服务多次业务请求
5. **可观测**：证书加载、握手、重载、过期均有 `tracing` 日志

### 1.4 使用场景

| 场景 | 路径 | 期望行为 |
|------|------|---------|
| 进程启动 | 读取 `TlsConfig` → 加载证书 → 启动 listener → 业务就绪 | 启动失败时 panic（证书配置错） |
| 业务调用 NaCRE | `create_sandbox()` → `client.request()` | 命中 keep-alive 池；30s 超时；失败返回错误 |
| NaCRE 沙箱回调 | 沙箱发 mTLS 请求 → listener 接收 → TLS 握手 → 处理注册 | 验证沙箱证书；处理后回调 `tx` |
| 证书到期/轮换 | 运维替换文件 → Unix socket 发 `TlsReload` | 0 秒内新握手走新证书；进行中请求不中断 |
| NaCRE 无响应 | `client.request()` 永久挂起 | 30s 后超时返回，释放读锁 |
| 私钥泄露 | 运维通过 `TlsReload` 通道换新证书 | 攻击者的旧证书无法再通过 mTLS 验证 |

---

## 二、详细设计

### 2.1 架构概览

证书管理涉及 **5 个文件的协作**：

```
src/
├── config/router.rs                  → TlsConfig 结构（文件路径配置）
├── scc/mod.rs                        → scc::decrypt 私钥解密
├── sandbox/
│   ├── nacre_client.rs               → 客户端 (outbound, 调 NaCRE)
│   ├── nacre_listener.rs             → 服务端 (inbound, 沙箱回调)
│   └── cert_resolver.rs              → 服务端动态证书解析器
├── notification/tls_subscriber.rs    → 证书重载通知入口
└── sandbox/sandbox_manager.rs        → 协调所有组件
```

**组件关系**：

```
                              ┌─────────────────────────────┐
                              │      SandboxManager         │
                              │  nacre_client: RwLock<...>  │
                              │  nacre_listener: RwLock<...> │
                              └────────┬──────────┬─────────┘
                                       │          │
                       ┌───────────────┘          └────────────────┐
                       ▼                                            ▼
              ┌─────────────────┐                       ┌──────────────────────┐
              │   NacreClient   │                       │    NacreListener     │
              │  (outbound)     │                       │  (inbound server)    │
              │                 │                       │                      │
              │  - tls_config   │                       │  - tls_config        │
              │  - cert/key     │                       │  - cert_resolver     │
              │  - ca_store Arc │                       │      (DynamicCert)   │
              │  - request_to   │                       │  - acceptor          │
              │    timeout      │                       │      (ServerConfig)  │
              │  - OnceCell     │                       │                      │
              │    <Client>     │                       └─────────┬────────────┘
              └─────────────────┘                                 │
                       │                                           │
                       │  tokio::time::timeout                     │  ResolvesServerCert
                       │  hyper-util::Client                       │  trait impl
                       │  HttpsConnector                           ▼
                       │                                  ┌──────────────────────┐
                       │                                  │ DynamicCertResolver  │
                       │                                  │  Arc<RwLock<         │
                       │                                  │    Option<Arc<       │
                       │                                  │      CertifiedKey>    │
                       │                                  │    >>>               │
                       │                                  └──────────────────────┘
                       │
                       ▼
              ┌─────────────────┐         ┌──────────────────┐
              │  Nacre Server   │         │  TlsReload       │
              │  (TLS endpoint) │         │  Subscriber      │
              └─────────────────┘         │  (Unix socket)   │
                                           └──────────────────┘
```

### 2.2 配置层（TlsConfig）

```rust
// src/config/router.rs
pub struct TlsConfig {
    pub enabled: bool,         // 总开关
    pub ca_cert_path: String,  // CA 证书（用于验证对端）
    pub cert_path: String,     // 自己的证书
    pub key_path: String,      // 自己的私钥（加密存储）
}
```

**两份 TlsConfig 实例**：

| 实例 | 配置路径 | 服务方向 |
|------|---------|---------|
| 客户端 TlsConfig | `sandbox_proxy.nacre.tls` | 我们调 NaCRE |
| 服务端 TlsConfig | `sandbox_proxy.listeners.tls` | 沙箱回调我们 |

**配置示例**（YAML）：
```yaml
sandbox_proxy:
  nacre:
    tls:
      enabled: true
      ca_cert_path: /etc/intellirouter/certs/nacre-ca.pem
      cert_path: /etc/intellirouter/certs/client.pem
      key_path: /etc/intellirouter/certs/client-key.pem
    client_config:
      timeout: 30000            # 单个 HTTP 请求超时（毫秒）
      registration_timeout: 180
      ...
  listeners:
    tls:
      enabled: true
      ca_cert_path: /etc/intellirouter/certs/sandbox-ca.pem
      cert_path: /etc/intellirouter/certs/server.pem
      key_path: /etc/intellirouter/certs/server-key.pem
    address: "0.0.0.0"
    port: 8443
```

### 2.3 磁盘存储

**证书文件格式**：

| 文件 | 内容 | 加密 | 格式 |
|------|------|------|------|
| `cert.pem` | 证书链（CA → 我们的证书） | 明文 | PEM |
| `key.pem` | 私钥 | **加密** | 加密二进制 |
| `*-ca.pem` | 信任的 CA 证书 | 明文 | PEM |

**为什么私钥加密存储**：
- 防运维误操作导致文件泄露时私钥直接暴露
- 即使有人拿到磁盘镜像，没有 SCC 密钥也解不开
- 防御深度：磁盘加密只是其中一层

### 2.4 SCC 解密层

```rust
// src/scc/mod.rs
pub fn decrypt(ciphertext: &[u8]) -> SCCResult<Zeroizing<Vec<u8>>> {
    // TODO: 联调之后接入 SCC_Decrypt
    Ok(Zeroizing::new(ciphertext.to_vec()))
}
```

**当前状态**：stub。联调后会接入真实的 SCC FFI（`SCC_Decrypt`）。

**`Zeroizing<Vec<u8>>`**：保证内存 drop 时自动清零，防内存取证。

**关键约束**：
- `scc::decrypt` 的输出**必须**是 `Zeroizing<Vec<u8>>` 类型
- 所有持有解密后私钥的变量都应该用此类型
- 不可绕过 SCC 自行读取明文私钥

### 2.5 客户端证书加载（NacreClient）

#### 2.5.1 数据结构

```rust
// src/sandbox/nacre_client.rs
pub struct NacreClient {
    tls_config: TlsConfig,

    // 缓存的证书（启动时 / reload 时填充）
    certificate: Option<CertificateDer<'static>>,         // 第一个证书
    private_key_bytes: Option<Zeroizing<Vec<u8>>>,        // 解密后的私钥字节（PEM 格式）
    ca_certificates: Option<Arc<rustls::RootCertStore>>,  // CA 证书池

    // 请求超时（来自 NacreClientConfig.timeout）
    request_timeout: Duration,

    // 懒加载的 Client 实例（连接池 + TLS 配置）
    http_client: OnceCell<Client<HttpConnector, Full<Bytes>>>,
    https_client: OnceCell<Client<HttpsConnector<HttpConnector>, Full<Bytes>>>,
}
```

**关键设计**：
- `ca_certificates` 套 `Arc` —— 共享而非深拷贝整个 cert store
- `private_key_bytes` 保留**原始 PEM 字节**（不是解析后的 `PrivateKeyDer`）—— 便于证书重载时重新解析
- `request_timeout` 由 `NacreClientConfig.timeout` 传入，贯穿到所有 `client.request()`
- `http_client` / `https_client` 用 `OnceCell` 缓存，**懒加载 + 跨请求复用**

#### 2.5.2 加载流程

```rust
fn load_certificates(&mut self) -> Result<()> {
    // 1. 加载 CA 证书（验证对端用）
    if self.ca_certificates.is_none() {
        let mut root_store = rustls::RootCertStore::empty();
        if !tls_config.ca_cert_path.is_empty() {
            // 从文件读 CA 证书
            let certs: Vec<CertificateDer> = rustls_pemfile::certs(&mut ca_cert_reader)...;
            for cert in certs { root_store.add(cert)?; }
        } else {
            // 用系统 CA 池
            rustls_native_certs::load_native_certs()...;
        }
        self.ca_certificates = Some(Arc::new(root_store));
    }

    // 2. 加载自己的证书 + 私钥（mTLS 用）
    if !tls_config.cert_path.is_empty() && !tls_config.key_path.is_empty() {
        // 读明文证书
        let certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut cert_reader)...;

        // 读加密私钥 → SCC 解密 → Zeroizing<Vec<u8>>
        let decrypted_key: Zeroizing<Vec<u8>> = scc::decrypt(&key_data)?;

        // 存进字段（PEM 字节，等 build_https_client 时再解析）
        self.certificate = Some(certs.into_iter().next().ok_or(...)?);
        self.private_key_bytes = Some(decrypted_key);
    }
    Ok(())
}
```

**为什么私钥先存字节不直接解析**：
- `PrivateKeyDer<'a>` 是借用类型，引用 `Zeroizing<Vec<u8>>` 的内存
- `OnceCell` 缓存的 `Client` 是 `'static` 的，不能借用字段
- 等真正 `build_https_client()` 时调 `.clone_key()` 转 `'static` owned key

#### 2.5.3 Client 构建（懒加载）

```rust
fn build_https_client(&self) -> Result<Client<...>> {
    // 1. 用缓存的 cert + key 构造 rustls::ClientConfig
    let key = private_key(&mut key_bytes[..])?
        .ok_or(...)?
        .clone_key();  // ← 关键：转 'static owned key

    let config = rustls::ClientConfig::builder()
        .with_root_certificates((*ca_certificates).clone())  // Arc 解引用 + 一次性深拷贝
        .with_client_auth_cert(vec![cert], key)
        .context("Failed to configure mTLS")?;

    // 2. 构造 HttpsConnector
    let connector = hyper_rustls::HttpsConnectorBuilder::new()
        .with_tls_config(config)
        .https_or_http()
        .enable_http1()
        .build();

    // 3. 构造 Client（自带连接池）
    Ok(Client::builder(TokioExecutor::new())
        .pool_idle_timeout(CLIENT_POOL_IDLE_TIMEOUT)  // 90s
        .build(connector))
}

async fn get_https_client(&self) -> Result<&Client<...>> {
    self.https_client
        .get_or_try_init(|| async { self.build_https_client() })
        .await
}
```

**`pool_idle_timeout(90s)` 的意义**：
- hyper-util `Client` 默认永久保留 idle 连接
- 90s 覆盖大多数企业 NAT 设备的会话超时
- 避免「连接已死」造成「Connection reset by peer」

**`(*ca_certificates).clone()` 的一次性成本**：
- rustls builder API 要求 `RootCertStore` by-value
- `*ca_certificates` 解引用 Arc 拿到 `RootCertStore`，`.clone()` 深拷贝
- **这步拷贝只发生一次**（build_https_client 时），OnceCell 缓存后不再发生

#### 2.5.4 请求发起

```rust
pub async fn create_sandbox(&self, target_url: &str, request: &CreateSandboxRequest) -> Result<NacreResponse> {
    // ... build http_request ...

    let (status, response_body) = if is_https {
        let client = self.get_https_client().await?;  // 拿到缓存的 &Client
        let response = tokio::time::timeout(
            self.request_timeout,                      // 来自 NacreClientConfig.timeout
            client.request(http_request)
        )
        .await
        .with_context(|| format!("Create sandbox HTTPS request timed out after {:?}", self.request_timeout))?
        .context("Failed to send create sandbox request (HTTPS)")?;
        // ... 读 body ...
    } else {
        // HTTP 分支同样模式
    };
    Ok(NacreResponse::new(status.as_u16(), response_body))
}
```

**关键点**：
- `get_https_client().await` 第一次调触发 build，后续直接返回 `&Client`（O(1)）
- `tokio::time::timeout` 保证请求有上限时间
- **超时触发后立即释放 `&self` 借用**，上层 `RwLock::read().await` 跟着释放
- 这就是 SB-DEV-003 修复的核心：保护写锁（证书重载）不被永久阻塞

### 2.6 服务端证书加载（NacreListener + DynamicCertResolver）

#### 2.6.1 NacreListener 结构

```rust
pub struct NacreListener {
    registration_tx: mpsc::UnboundedSender<SandboxRegistrationEvent>,
    registration_rx: Option<mpsc::UnboundedReceiver<SandboxRegistrationEvent>>,
    server_handle: Option<tokio::task::JoinHandle<()>>,
    tls_config: TlsConfig,
    session_cache: Arc<SessionCache>,
    cert_resolver: Arc<DynamicCertResolver>,  // ← 关键：动态证书解析器
}
```

**`cert_resolver` 是 `Arc`**：可以 clone 多个引用，共享同一份证书状态。

#### 2.6.2 DynamicCertResolver

```rust
// src/sandbox/cert_resolver.rs
pub struct DynamicCertResolver {
    certified_key: Arc<RwLock<Option<Arc<rustls::sign::CertifiedKey>>>>,
}

impl DynamicCertResolver {
    pub fn update_certificate(&self, cert_chain: Vec<CertificateDer<'static>>,
        private_key: &PrivateKeyDer<'static>) -> Result<()> {
        let signing_key = rustls::crypto::aws_lc_rs::sign::any_supported_type(&private_key)?;
        let certified_key = CertifiedKey::new(cert_chain, signing_key);
        let mut guard = self.certified_key.write()?;
        *guard = Some(Arc::new(certified_key));  // ← 关键：原地换 Arc 指针
        Ok(())
    }

    pub fn clear_certificate(&self) -> Result<()> {
        let mut guard = self.certified_key.write()?;
        *guard = None;
        Ok(())
    }
}

impl ResolvesServerCert for DynamicCertResolver {
    fn resolve(&self, _client_hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        match self.certified_key.read() {
            Ok(guard) => guard.clone(),
            Err(_) => None,
        }
    }
}
```

**精妙之处**：
- `update_certificate` 只是 `RwLock.write()` 换 `Option` 里的 `Arc<CertifiedKey>`
- 正在握手的请求持有**旧 Arc 引用**（rustls 的设计），不会被打断
- 新握手调 `resolve()` 从 `Option` 拿到**新证书**
- **零重启热更新**

#### 2.6.3 加载流程

```rust
fn load_certificates(&mut self) -> Result<()> {
    // 1. 读明文证书链
    let certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut cert_reader)...;

    // 2. 读加密私钥 → SCC 解密
    let decrypted_key: Zeroizing<Vec<u8>> = scc::decrypt(&key_data)?;

    // 3. 解析私钥（PEM → PrivateKeyDer）
    let mut key_reader = &decrypted_key[..];
    let mut private_key = rustls_pemfile::private_key(&mut key_reader)?
        .ok_or_else(|| anyhow::anyhow!("No private key found"))?;

    // 4. 关键：塞进 DynamicCertResolver（不是直接传给 ServerConfig）
    let ret = self.cert_resolver.update_certificate(certs, &private_key);
    private_key.zeroize();  // 显式清零解析后的私钥
    ret
}
```

**`private_key.zeroize()`**：解析后的 `PrivateKeyDer` 也用 `Zeroize`，防止遗留内存。

#### 2.6.4 ServerConfig 构建（启动时一次性）

```rust
// 在 start() 中
let config = if !self.tls_config.ca_cert_path.is_empty() {
    // mTLS：也要验证客户端证书
    let ca_certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut ca_cert_reader)...;
    let mut root_store = rustls::RootCertStore::empty();
    for cert in ca_certs { root_store.add(cert)?; }
    let client_cert_verifier = WebPkiClientVerifier::builder(Arc::new(root_store))
        .build()?;
    ServerConfig::builder()
        .with_client_cert_verifier(client_cert_verifier)
        .with_cert_resolver(self.cert_resolver.clone())  // ← 关键：引用永不变
} else {
    // 单向 TLS
    ServerConfig::builder()
        .with_no_client_auth()
        .with_cert_resolver(self.cert_resolver.clone())
};
let acceptor = TlsAcceptor::from(Arc::new(config));
```

**为什么 `with_cert_resolver(self.cert_resolver.clone())`**：
- `cert_resolver: Arc<DynamicCertResolver>` 引用克隆便宜（O(1) 引用计数）
- ServerConfig 持有的是 `Arc<dyn ResolvesServerCert>` trait object
- **整个进程生命周期内**这个引用不变
- 证书热更新时只换 `cert_resolver` 内部的 `Arc<CertifiedKey>`，**不重建 ServerConfig**

#### 2.6.5 接收连接

```rust
loop {
    let (stream, _) = listener.accept().await?;
    let acceptor = acceptor.clone();
    tokio::spawn(async move {
        let tls_stream = acceptor.accept(stream).await?;  // ← TLS 握手在这里
        // ... 处理 HTTP 请求 ...
    });
}
```

**每个连接 spawn 一个 task**：
- TLS 握手在子 task 中进行
- acceptor 可 clone（内部是 Arc），多个 task 共享
- 不影响 listener accept 循环

### 2.7 TLS 握手

#### 2.7.1 客户端 mTLS 握手

```
Client                                                  Server (NaCRE)
  │                                                          │
  ├─ ClientHello ──────────────────────────────────────────►│
  │                                                          │
  │◄── ServerHello + ServerCertChain ──────────────────────┤
  │   + CertificateRequest (要求客户端发证书) ──────────────┤
  │                                                          │
  ├─ ClientCertChain ─────────────────────────────────────►│  ← 我们的证书
  │   + ClientKeyExchange ────────────────────────────────►│
  │   + CertificateVerify (用我们的私钥签名) ──────────────►│  ← 证明持有该证书
  │                                                          │
  │◄── Finished ────────────────────────────────────────────┤
  │                                                          │
  ├─ Finished ─────────────────────────────────────────────►│
  │                                                          │
  │         Encrypted application data                      │
```

**两方向验证**：
- **客户端验证服务端**：`ClientConfig::with_root_certificates(ca_certificates)` —— 服务端证书必须由这些 CA 签发
- **服务端验证客户端**：`ClientConfig::with_client_auth_cert(cert, key)` —— 握手时把我们的证书 + 私钥签名发给服务端

#### 2.7.2 服务端 mTLS 握手

```rust
// TlsAcceptor 内部
fn accept(&self, stream: TcpStream) -> Result<TlsStream> {
    // 1. 接收 ClientHello
    // 2. 调 cert_resolver.resolve(client_hello) 拿服务端证书
    let server_cert = self.cert_resolver.resolve(client_hello)?;
    // 3. 发送 ServerHello + ServerCertChain
    // 4. 接收 ClientCertChain → 用 WebPkiClientVerifier 验证
    self.client_cert_verifier.verify_client_cert(client_chain, roots)?;
    // 5. 验证 CertificateVerify (客户端用私钥签名)
    // 6. 交换 Finished
    Ok(...)
}
```

**`resolve()` 每次握手调一次**：
- `RwLock.read()` 取当前 `Arc<CertifiedKey>`
- 零缓存开销，新证书**立即生效**
- 旧 Arc 引用被正在进行的握手持有，不会被 drop

#### 2.7.3 性能特征

| 操作 | 典型耗时 | 是否每次请求都做 |
|------|---------|----------------|
| TCP 握手 | 0.5-1 RTT | 新连接时 |
| TLS 握手（mTLS） | 1-2 RTT | 新连接时 |
| HTTP 请求/响应 | 业务处理 | 每次 |
| **复用 keep-alive** | **直接发 HTTP** | **第二请求起** |

典型企业网络 RTT ~50ms，单次 mTLS 握手 ~100-200ms，**复用后 1-5ms 完成请求**。

### 2.8 证书重载（核心）

#### 2.8.1 触发入口

```rust
// src/main.rs
let tls_subscriber = TlsReloadSubscriber::new(
    config.sandbox_proxy.nacre.tls.clone(),
    config.sandbox_proxy.listeners.tls.clone(),
);
notification_server.add_subscriber(Arc::new(tls_subscriber)).await;
```

**NotificationServer** 监听 Unix Domain Socket（`notification.socket_path`），外部工具发 `TlsReload` 消息触发。

**`TlsReloadSubscriber` 内部保存最新 TlsConfig**：消息体不传任何参数，重载用的是订阅者内部的配置副本。

#### 2.8.2 通知分发

```rust
// src/notification/tls_subscriber.rs
impl NotificationSubscriber for TlsReloadSubscriber {
    fn interested_types(&self) -> Vec<NotificationType> {
        vec![NotificationType::TlsReload]
    }

    async fn on_notification(&self, _message: &NotificationMessage) -> anyhow::Result<()> {
        let sandbox_manager = SandboxManager::get()?;
        let client_result = sandbox_manager
            .reload_client_certificates(self.nacre_tls_config.clone())
            .await;
        let listener_result = sandbox_manager
            .reload_listener_certificates(self.listeners_tls_config.clone())
            .await;
        // ...
    }
}
```

**两步独立重载**：客户端 + 服务端，任一失败返回错误。

#### 2.8.3 客户端重载

```rust
// src/sandbox/nacre_client.rs
pub fn reload_tls_config(&mut self, tls_config: TlsConfig) -> Result<()> {
    self.tls_config = tls_config;

    // 1. 清空所有缓存字段
    self.certificate = None;
    self.private_key_bytes = None;       // Zeroizing drop 时清零内存
    self.ca_certificates = None;          // Arc drop 时引用计数 -1

    // 2. ★ 关键：清空 OnceCell 让旧 Client drop ★
    self.http_client = OnceCell::new();
    self.https_client = OnceCell::new();

    // 3. 重新加载证书
    if self.tls_config.enabled {
        self.load_certificates()?;
    }
    Ok(())
}
```

**为什么必须清 `OnceCell`**：
- 旧 `Client` 内部的 `HttpsConnector` 引用旧 `rustls::ClientConfig`（旧证书 + 旧私钥）
- 不清的话，下次请求仍用旧证书握手 → 失败

**清空过程**：
1. `self.https_client = OnceCell::new()` → 旧 `OnceCell` 变量 drop
2. 旧 `Client` 引用计数 -1
3. 若无正在用的 `&Client` 引用（重载时通常业务请求都结束了）→ 引用归零
4. `Client` drop → 内部连接池的所有 idle 连接优雅关闭（TCP FIN）
5. `HttpsConnector` drop → 释放 TLS 配置

**安全保证**：Rust 借用系统保证 `&Client` 引用有效期内 `Client` 不会 drop。

#### 2.8.4 服务端重载

```rust
// src/sandbox/nacre_listener.rs
pub fn reload_tls_config(&mut self, tls_config: TlsConfig) -> Result<()> {
    self.tls_config = tls_config;
    if self.tls_config.enabled {
        self.load_certificates()?;  // → cert_resolver.update_certificate()
    } else {
        self.cert_resolver.clear_certificate()?;
    }
    Ok(())
}
```

**`load_certificates` 内部**调 `cert_resolver.update_certificate(certs, &key)`，在 `RwLock.write()` 保护下替换 `Arc<CertifiedKey>`。

**与客户端的关键差异**：
- 服务端**不重建**任何东西
- `ServerConfig::cert_resolver: Arc<DynamicCertResolver>` 引用永不变
- `TlsAcceptor` / listener 进程**完全不重启**
- 零停机，0 业务中断

#### 2.8.5 客户端 vs 服务端：两种热重载模式对比

| 维度 | 客户端 (NacreClient) | 服务端 (NacreListener) |
|------|---------------------|----------------------|
| 重载方式 | 清 OnceCell → 重建 Client | 改 cert_resolver 内部 Arc |
| ServerConfig 重建 | 是 | 否 |
| 进行中请求 | 持有旧 `&Client`，正常完成 | 持有旧 `Arc<CertifiedKey>`，正常完成 |
| 新请求路径 | 走 `get_or_init` 拿到新 Client | 调 `cert_resolver.resolve()` 拿新证书 |
| 停机时间 | 0 秒 | 0 秒 |
| 根本原因 | hyper `Client<C, B>` 泛型设计 | rustls `Arc<dyn ResolvesServerCert>` trait object |

**为什么客户端不能也用 cert_resolver 方式**：
```rust
// hyper::Client<C, B> 是泛型
pub struct Client<C, B> {
    connector: C,  // C 是泛型参数
}
// 想换 connector 必须换 Client 实例
```

hyper 的设计**不支持**运行时换 connector，所以客户端必须重建。

#### 2.8.6 RwLock 写锁保护（SB-DEV-003 修复点）

```rust
// src/sandbox/sandbox_manager.rs
pub async fn reload_client_certificates(&self, tls_config: TlsConfig) -> Result<()> {
    self.nacre_client.write().await     // ← 写锁
        .reload_tls_config(tls_config)
        .context("Failed to reload client TLS certificates")
}
```

**问题**：业务请求 `create_sandbox` 时持有 `nacre_client.read().await` 读锁。

```
修复前:
  业务线程 A: read().await 拿读锁 → client.request() 永远不返回
  运维触发重载: write().await 永远等读锁释放 → ❌ 卡死

修复后:
  业务线程 A: read().await → client.request() 在 30s 后超时 → 读锁释放
  运维触发重载: write().await 拿到写锁 → ✅ 重载成功
```

`tokio::time::timeout(self.request_timeout, ...)` 是关键。

### 2.9 客户端连接复用

#### 2.9.1 双重缓存机制

| 层级 | 机制 | 提供者 |
|------|------|--------|
| 第一层 | `Client` 实例被缓存 | `OnceCell` |
| 第二层 | 单个 `Client` 内部维护 idle keep-alive 连接池 | `hyper_util` |

**两层缺一不可**：
- 缺第一层：每次新建 `Client`，连接池随之销毁 → 0 复用
- 缺第二层：`Client` 内部不复用 → 每次请求都新建连接

#### 2.9.2 性能数据估算（典型 mTLS 环境）

假设 50 QPS 持续请求 NaCRE，单沙箱操作 5ms 业务处理：

| 场景 | 单请求延迟 | 50 QPS 总耗时/s |
|------|-----------|----------------|
| **修复前**（每次新建连接 + 每次深拷贝 cert store） | ~150ms | 50 × 150ms = **7.5s/s**（超载） |
| **修复后**（复用 + 共享） | ~5-10ms | 50 × 10ms = **500ms/s**（游刃有余） |
| **提升** | **15-30x** | **15x** |

#### 2.9.3 keep-alive 超时选择（90s）

- 大多数企业 NAT 设备的会话超时：60-120s
- 取 90s 留 buffer
- 比 60s 安全（避免误杀），比 300s 节省内存

**hyper-util 内部行为**：
- 懒检查：每次 `get_idle_connection` 时检查是否过期
- 无后台定时器主动清理
- 过期连接在**下一次请求**时才被移除

#### 2.9.4 完整请求时序（带连接复用）

```
T=0ms       业务请求 → OnceCell 冷启动 → build_https_client() → 30ms 完成
T=30ms      client.request() → 池空 → TCP + TLS 握手 → 230ms 完成
            idle 连接进池

T=1000ms    业务请求 → OnceCell 热路径 → 拿 &Client (O(1))
            client.request() → 命中池中 idle 连接 → 30ms 完成
            (跳过了 200ms 的 TCP+TLS 握手)

T=92000ms   业务请求 → 池中连接 idle 90s → 过期丢弃
            client.request() → 走冷启动 → 230ms 完成
```

---

## 三、关键流程时序图

### 3.1 启动加载

```
YAML 配置 (TlsConfig)
   │
   ▼
SandboxManager::new()
   │
   ├─ NacreClient::new(tls_config, app_code, request_timeout)
   │    │
   │    ├─ 字段初始化 (OnceCell 空, request_timeout, etc.)
   │    └─ load_certificates() [如果 enabled]
   │         ├─ 读 ca_cert_path → Arc<RootCertStore>
   │         └─ 读 cert + scc::decrypt(key) → (cert, key_bytes)
   │
   └─ NacreListener::new(tls_config, session_cache)
        │
        ├─ 创建空 DynamicCertResolver
        └─ (start() 时才 load_certificates)

NacreListener::start()
   │
   ├─ load_certificates() → cert_resolver.update_certificate(certs, key)
   │
   ├─ 构造 ServerConfig (with_cert_resolver(cert_resolver.clone()))
   │
   ├─ TlsAcceptor::from(Arc::new(config))
   │
   └─ spawn listener 循环
        │
        └─ 每连接: acceptor.accept(stream) → TLS 握手 → 处理 HTTP
```

### 3.2 业务请求 (create_sandbox)

```
业务线程
   │
   ▼
SandboxManager::create_sandbox()
   │
   ├─ nacre_client.read().await     ← 拿读锁
   │
   ├─ NacreClient::create_sandbox()
   │    │
   │    ├─ build http_request
   │    │
   │    ├─ get_https_client().await
   │    │    │
   │    │    ├─ 首次: build_https_client() → OnceCell.set
   │    │    └─ 后续: 直接返回 &
   │    │
   │    ├─ tokio::time::timeout(30s, client.request(req))
   │    │    │
   │    │    ├─ 30s 内响应: Ok(response)
   │    │    └─ 30s 后: Err(超时) → future drop → 读锁释放
   │    │
   │    └─ 读 response body → 构造 NacreResponse
   │
   └─ 读锁释放
```

### 3.3 证书重载

```
外部运维
   │
   ├─ 替换 /path/to/cert.pem 和 /path/to/key.pem
   │
   └─ 通过 Unix socket 发 "TlsReload"
        │
        ▼
TlsReloadSubscriber::on_notification
   │
   ├─ SandboxManager::reload_client_certificates(new)
   │    │
   │    ├─ nacre_client.write().await   ← 写锁等待读锁
   │    │                                  ⚠️ 业务读锁必须先释放
   │    │
   │    └─ NacreClient::reload_tls_config(new)
   │         │
   │         ├─ 清字段 (cert, key_bytes, ca_certificates)
   │         │
   │         ├─ 清 OnceCell (http_client, https_client)
   │         │    └─ 旧 Client drop → 关闭 idle 连接
   │         │
   │         └─ load_certificates() → 读新文件
   │
   └─ SandboxManager::reload_listener_certificates(new)
        │
        ├─ nacre_listener.write().await
        │
        └─ NacreListener::reload_tls_config(new)
             │
             └─ cert_resolver.update_certificate(new_certs, new_key)
                  └─ RwLock.write() 换 Arc<CertifiedKey>
                  (零重启热更新)

T7
业务请求 N+1
   │
   ├─ get_https_client() 冷启动 (OnceCell 空)
   │    └─ build_https_client() 用新字段构造新 Client
   │
   └─ client.request() → 走新连接 (mTLS 用新证书)

T8
新 TLS 握手到达 NacreListener
   │
   └─ acceptor.accept(stream)
        └─ cert_resolver.resolve() → 拿到新 Arc<CertifiedKey> → 新证书
```

---

## 四、可靠性设计

### 4.1 SB-DEV-003 修复：HTTP 请求超时

**问题描述**：
- 客户端 `client.request()` 无超时保护
- NaCRE 无响应时请求永久挂起
- 持有 `nacre_client` 读锁，阻塞证书重载等写锁

**修复**：
- `NacreClient` 加 `request_timeout: Duration` 字段
- 4 处 `client.request(http_request).await` 全部用 `tokio::time::timeout` 包裹
- `Client::builder()` 加 `.pool_idle_timeout(90s)` 作为第二道防线
- 调用方 `sandbox_manager.rs` 构造 `NacreClient` 时传入 `Duration::from_millis(config.nacre.client_config.timeout)`

**验证**：
- `test_create_sandbox_times_out_for_unreachable_host` —— 200ms 超时 + RFC 5737 不可达地址 192.0.2.1，验证请求在超时时间内返回

### 4.2 SB-DEV-004 优化：连接池复用

**问题描述**：
- 每次请求新建 `Client` + TLS 握手
- mTLS + RSA 4096 下单次操作 ~100-300ms
- 同时每次 `ca_certificates.clone()` 深拷贝整个 cert store

**修复**：
- `NacreClient` 加 `http_client: OnceCell<...>` 和 `https_client: OnceCell<...>` 缓存 `Client`
- 新增 `get_http_client()` / `get_https_client()` 懒加载方法
- `ca_certificates` 改为 `Option<Arc<RootCertStore>>` 共享
- 私钥解析用 `.clone_key()` 转 `'static` owned key
- `reload_tls_config` 时清空 `OnceCell` 触发旧 Client drop

**验证**：
- `test_http_client_is_cached_across_calls` —— `std::ptr::eq` 验证两次 get 拿同一对象

### 4.3 超时与 RwLock 的耦合

**核心保护**：
- 业务请求加超时 → 读锁必有释放时间
- 写锁可获得 → 证书重载不被卡死
- **零停机证书轮换**才有可能

**配置**：
- `NacreClientConfig.timeout` 默认 30000ms
- YAML: `sandbox_proxy.nacre.client_config.timeout`
- 类型: `u64` (毫秒) → `Duration::from_millis(...)`

---

## 五、安全设计

### 5.1 私钥加密存储

| 阶段 | 状态 | 防护 |
|------|------|------|
| 磁盘 | 加密二进制 | SCC 加密 |
| 加载到内存 | `Zeroizing<Vec<u8>>`（PEM 字节） | 显式清零 |
| 解析使用 | `PrivateKeyDer<'static>` | 范围限定在 `OnceCell<Client>` 内 |
| 使用完毕 | rustls 内部用完即 drop | `Zeroizing` 自动清零 |
| 备份 | 应另行加密 | 不在本项目控制范围 |

**注意**：`scc::decrypt` 当前是 stub，联调后必须接入真实 SCC。

### 5.2 内存安全

- `private_key_bytes: Option<Zeroizing<Vec<u8>>>` —— 显式 `Zeroize` trait
- `ca_certificates: Option<Arc<...>>` —— 多份引用共享，drop 时引用计数归零才释放
- `OnceCell` 缓存的 `Client` 跨请求存在，但 `HttpsConnector` 内部 TLS 配置自包含

### 5.3 证书泄露响应

#### 5.3.1 威胁模型

| 泄露级别 | 泄露内容 | 影响 |
|---------|---------|------|
| L1 | 磁盘上加密的 key.pem | 需要破解 SCC |
| L2 | 内存中解密后的私钥（需 root dump） | 中等风险 |
| L3 | 明文私钥 + 证书都被攻击者拿到 | 严重：可冒充身份 |

#### 5.3.2 mTLS 双向认证的局限

mTLS 只能防"伪造证书"，**不能防"合法证书被滥用"**：
- 攻击者拿到我们自己的证书 + 私钥
- Nacre 端**无法区分**是攻击者还是我们
- mTLS 握手会通过

**应用层第二道认证建议**：
- IP 白名单
- API Key / JWT
- 请求签名

#### 5.3.3 事件响应流程

| 步骤 | 操作 | 项目提供的支持 |
|------|------|---------------|
| 1. 检测 | 日志/告警/审计 | 依赖外部 SOC |
| 2. 隔离 | 把泄露指纹加入 CRL/OCSP | **依赖 Nacre 端** |
| 3. 生成新证书 | CA 申请 + SCC 加密 | (外部流程) |
| 4. 部署新证书 | 替换文件 + TlsReload | ✅ Unix socket 触发 |
| 5. 验证 | 健康检查 | 内置 |
| 6. 根因分析 | RCA | 依赖流程 |

**项目提供的关键能力**：
- **秒级证书轮换**：通过 TlsReload 通道，0 秒停机
- **mTLS 失败告警**：可在 listener 端记录所有握手失败
- **配置集中管理**：`TlsConfig` 是单一真相源

#### 5.3.4 已知不足

| 不足 | 风险 | 建议 |
|------|------|------|
| `scc::decrypt` 是 stub | "加密存储"是安慰剂 | 联调必须接入 |
| 无 CRL/OCSP 检查 | 攻击者仍可用旧证书 | Nacre 端实现 |
| Unix socket 无认证 | DoS：反复触发 reload | 加 token 认证、限流 |
| 无证书使用日志 | 难以溯源 | 记录 client cert fingerprint |
| 无证书生命周期管理 | 过期未告警 | 启动检查 + 续期 |
| 无应用层第二认证 | mTLS 失败时无兜底 | 加 IP 白名单/JWT |

---

## 六、API 参考

### 6.1 NacreClient

```rust
impl NacreClient {
    /// 构造（懒加载，不立即构建 Client）
    pub fn new(tls_config: TlsConfig, app_code: &str, request_timeout: Duration) -> Result<Self>;

    /// 热重载证书（清空缓存，重新加载）
    pub fn reload_tls_config(&mut self, tls_config: TlsConfig) -> Result<()>;

    /// 创建沙箱（POST + JSON body）
    pub async fn create_sandbox(&self, target_url: &str, request: &CreateSandboxRequest) -> Result<NacreResponse>;

    /// 删除沙箱（DELETE）
    pub async fn delete_sandbox(&self, target_url: &str) -> Result<StatusCode>;
}
```

### 6.2 NacreListener

```rust
impl NacreListener {
    /// 构造（创建空 cert_resolver）
    pub fn new(tls_config: TlsConfig, session_cache: Arc<SessionCache>) -> Result<Self>;

    /// 启动 HTTP/HTTPS 服务
    pub async fn start(&mut self, address: &str) -> Result<()>;

    /// 热重载证书（调 cert_resolver.update_certificate）
    pub fn reload_tls_config(&mut self, tls_config: TlsConfig) -> Result<()>;

    /// 取出注册事件接收器
    pub fn take_registration_rx(&mut self) -> Option<mpsc::UnboundedReceiver<SandboxRegistrationEvent>>;
}
```

### 6.3 SandboxManager (重载相关)

```rust
impl SandboxManager {
    /// 重载客户端证书
    pub async fn reload_client_certificates(&self, tls_config: TlsConfig) -> Result<()>;

    /// 重载服务端证书
    pub async fn reload_listener_certificates(&self, tls_config: TlsConfig) -> Result<()>;
}
```

### 6.4 TlsReloadSubscriber

```rust
impl NotificationSubscriber for TlsReloadSubscriber {
    fn name(&self) -> &str;                                    // "tls_reload"
    fn interested_types(&self) -> Vec<NotificationType>;        // [TlsReload]
    async fn on_notification(&self, _message: &NotificationMessage) -> anyhow::Result<()>;
}
```

---

## 七、测试

### 7.1 已有的单元测试

```rust
// src/sandbox/nacre_client.rs
#[test]
fn test_nacre_client_default() { /* 构造验证 */ }

#[tokio::test]
async fn test_http_client_is_cached_across_calls() {
    // 验证 OnceCell 缓存生效
    let c1 = client.get_http_client().await;
    let c2 = client.get_http_client().await;
    assert!(std::ptr::eq(c1, c2));
}

#[tokio::test]
async fn test_create_sandbox_times_out_for_unreachable_host() {
    // 验证 200ms 超时对 RFC 5737 保留地址 192.0.2.1 生效
    let start = std::time::Instant::now();
    let result = client.create_sandbox("http://192.0.2.1:65535/...", &request).await;
    let elapsed = start.elapsed();
    assert!(result.is_err());
    assert!(elapsed < Duration::from_secs(5));
}
```

### 7.2 端到端测试建议

| 场景 | 验证点 | 方法 |
|------|--------|------|
| 真实 mTLS 握手 | 客户端能连上真实 NaCRE | 集成测试（需要 Nacre 测试环境） |
| 证书重载 | 替换文件 + TlsReload → 下一个请求用新证书 | 集成测试 |
| 服务端 mTLS | 沙箱用合法证书能回调，用伪造证书被拒 | 集成测试 + 模拟沙箱 |
| 错误 CA | 客户端用错误 CA 拒绝服务端证书 | 单元测试（mock server） |
| 过期证书 | 客户端/服务端拒绝过期证书 | 时间 mock 单元测试 |

### 7.3 Chaos Engineering 建议

- 模拟 NaCRE 不响应 → 验证 30s 超时
- 模拟证书重载瞬间有进行中请求 → 验证请求不中断
- 模拟恶意 TlsReload 攻击 → 验证 Unix socket 防护（待实现）
- 模拟 SCC 解密失败 → 验证降级行为

---

## 八、未来优化方向

1. **CRL/OCSP 集成**：在 Nacre 端实现证书撤销检查
2. **证书使用日志**：记录 client cert fingerprint + 源 IP 到审计日志
3. **Unix socket 认证**：加 token / PID 校验防 DoS
4. **证书生命周期管理**：启动检查过期时间、提前告警
5. **应用层第二认证**：IP 白名单、API Key
6. **指标导出**：导出 TLS 握手次数、连接池大小、证书年龄等 Prometheus 指标
7. **自动化证书轮换**：集成 cert-manager 或 cfssl 自动签发
8. **HSM/TPM 集成**：私钥永不出硬件边界（成本较高）

---

## 九、变更历史

### v1.2.0 (2026-06-18)
- **修复 SB-DEV-003**：所有 `client.request()` 加 `tokio::time::timeout` 包裹，防止请求挂起阻塞证书重载
- **优化 SB-DEV-004**：客户端 `Client` 用 `OnceCell` 缓存复用，`ca_certificates` 改 `Arc<RootCertStore>`，私钥用 `.clone_key()` 转 `'static`
- 新增测试：`test_create_sandbox_times_out_for_unreachable_host`、`test_http_client_is_cached_across_calls`

### v1.1.0
- 引入 `DynamicCertResolver` 实现服务端证书热更新
- 引入 `TlsReloadSubscriber` 实现通知驱动的证书重载
- 引入 `scc::decrypt` 解密私钥（目前是 stub）

### v1.0.0
- 初始 mTLS 双向认证实现
- `NacreClient` / `NacreListener` 基本结构
- 启动时一次性加载证书
