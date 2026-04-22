# aiproxy

A model-aware, OpenAI-compatible reverse proxy for pools of local/remote LLM backends (llama.cpp, OpenAI, Anthropic, Google, Ollama).

> WARNING! This is still a work-in-progress. My main use-case is providing a single OpenAI API-compatible service from which I can reach multiple llama-server (llama.cpp) instances in my AI lab. It is working really well for this use-case so far. However I have not yet thoroughly tested the other backend LLM services. If you encounter any problems, please open an issue or PR.

## Supported backend `schema` values:

| Schema      | Description                                                                                                                                                                                                                                                |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `llamacpp`  | llama.cpp servers (OpenAI-compatible wire format, `/health` probe)                                                                                                                                                                                         |
| `openai`    | OpenAI API and OpenAI-compatible providers                                                                                                                                                                                                                 |
| `anthropic` | Anthropic Messages API (`/v1/messages`)                                                                                                                                                                                                                    |
| `google`    | Google Gemini API                                                                                                                                                                                                                                          |
| `ollama`    | Native Ollama API (`/api/chat`, `/api/embed`, `/api/tags`) with NDJSON streaming. Supports `chat_completions` and `embeddings`. `rerank` is unsupported. Ollama also exposes an OpenAI-compatible `/v1/*` shim, for which you can use the `openai` schema. |

## Supported Endpoints

| Endpoint         | OpenAI-compatible API  |
|------------------|------------------------|
| chat_completions | `/v1/chat/completions` |
| embeddings       | `/v1/embeddings`       |
| rerank           | `/v1/rerank`           |

## Example `config.yaml`

```yaml
server:
  host: '0.0.0.0'
  port: 8080
  api_key: '012345678901234567890123456789'
  log_level: 'info'  # debug, info, warn, error
  
  # TLS/HTTPS Configuration (optional)
  # Uncomment and configure to enable HTTPS
  tls:
    enabled: false  # Set to true to enable HTTPS
    # cert_file: '/path/to/server-cert.pem'     # Server certificate file
    # key_file: '/path/to/server-key.pem'       # Server private key file

pools:
  - model: 'qwen3.5-122b'
    endpoint: 'chat_completions' # chat_completions, embeddings, rerank
    schema: 'llamacpp' # anthropic, google, openai, llamacpp, ollama
    instances:
      - url: 'http://192.0.2.11:8080'
        api_key: 'abcdefabcdefabcdefabcdef'
        # tls:                                  # May be necessary if URL is https:
        #   ca_file: '/path/to/backend-ca.pem'  # Custom CA certificate for backend validation (use OS-installed CA certs if not specified)
        #   insecure_skip_verify: false         # Set to true to skip certificate verification (not recommended for production)
      - url: 'http://192.0.2.12:8080'
        api_key: 'abcdefabcdefabcdefabcdef'
        tls:
          ca_file: '/path/to/backend-ca.pem'
          insecure_skip_verify: false
    session_timeout: 300       # seconds
    health_check_interval: 30  # seconds

  - model: 'nemotron-embed-1b-v2'
    endpoint: 'embeddings' # chat_completions, embeddings, rerank
    schema: 'llamacpp' # anthropic, google, openai, llamacpp, ollama
    instances:
      - url: 'http://192.0.2.13:8080'
        api_key: 'fedcbafedcbafedcbafedcba'
    session_timeout: 300       # seconds
    health_check_interval: 30  # seconds
```

## Example `docker-compose.yaml`

```yaml
services:
  aiproxy:
    image: robcowart/aiproxy:latest
    container_name: aiproxy
    restart: unless-stopped
    volumes:
      - ./config.yaml:/etc/aiproxy/config.yaml:ro
    ports:
      - 8080:8080/tcp
```
