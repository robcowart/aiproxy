package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
	"go.uber.org/zap/zaptest/observer"
)

const validYAML = `
server:
  host: '0.0.0.0'
  port: 8080
  api_key: 'supersecretclientkey'
  log_level: 'info'
  tls:
    enabled: false
pools:
  - model: 'qwen3.5-122b'
    endpoint: 'chat_completions'
    schema: 'llamacpp'
    instances:
      - url: 'http://10.0.0.1:9999'
        api_key: 'backendkey1234'
    session_timeout: 300
    health_check_interval: 30
  - model: 'bge-reranker'
    endpoint: 'rerank'
    schema: 'llamacpp'
    instances:
      - url: 'http://10.0.0.2:9999'
        api_key: 'backendkey5678'
    session_timeout: 60
    health_check_interval: 15
`

func writeTempConfig(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "config.yaml")
	assert.NoError(t, os.WriteFile(p, []byte(body), 0o600))
	return p
}

func TestLoad_Valid(t *testing.T) {
	p := writeTempConfig(t, validYAML)
	cfg, err := Load(p)
	assert.NoError(t, err)
	assert.Equal(t, 8080, cfg.Server.Port)
	assert.Len(t, cfg.Pools, 2)
	assert.Equal(t, SchemaLlamaCPP, cfg.Pools[0].Schema)
	assert.Equal(t, EndpointChatCompletions, cfg.Pools[0].Endpoint)
	assert.Equal(t, 300, cfg.Pools[0].SessionTimeout)
}

func TestLoad_EnvOverride(t *testing.T) {
	p := writeTempConfig(t, validYAML)
	t.Setenv("AIPROXY_SERVER__PORT", "9091")
	cfg, err := Load(p)
	assert.NoError(t, err)
	assert.Equal(t, 9091, cfg.Server.Port)
}

func TestLoad_MissingAPIKey(t *testing.T) {
	y := `
server:
  host: '0.0.0.0'
  port: 8080
  log_level: 'info'
pools:
  - model: 'x'
    endpoint: 'chat_completions'
    schema: 'llamacpp'
    instances:
      - url: 'http://127.0.0.1:1'
        api_key: 'k'
`
	_, err := Load(writeTempConfig(t, y))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "api_key")
}

func TestLoad_DuplicateModel(t *testing.T) {
	y := `
server:
  port: 8080
  api_key: 'x'
pools:
  - model: 'a'
    endpoint: 'chat_completions'
    schema: 'llamacpp'
    instances: [{ url: 'http://x:1', api_key: 'k' }]
  - model: 'a'
    endpoint: 'chat_completions'
    schema: 'llamacpp'
    instances: [{ url: 'http://x:2', api_key: 'k' }]
`
	_, err := Load(writeTempConfig(t, y))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate")
}

func TestLoad_OllamaSchema(t *testing.T) {
	y := `
server:
  port: 8080
  api_key: 'x'
pools:
  - model: 'llama3.2'
    endpoint: 'chat_completions'
    schema: 'ollama'
    instances: [{ url: 'http://x:11434', api_key: 'k' }]
`
	cfg, err := Load(writeTempConfig(t, y))
	assert.NoError(t, err)
	assert.Len(t, cfg.Pools, 1)
	assert.Equal(t, SchemaOllama, cfg.Pools[0].Schema)
}

func TestLoad_InvalidSchema(t *testing.T) {
	y := `
server:
  port: 8080
  api_key: 'x'
pools:
  - model: 'a'
    endpoint: 'chat_completions'
    schema: 'nope'
    instances: [{ url: 'http://x:1', api_key: 'k' }]
`
	_, err := Load(writeTempConfig(t, y))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "schema")
}

func TestLoad_InvalidURL(t *testing.T) {
	y := `
server:
  port: 8080
  api_key: 'x'
pools:
  - model: 'a'
    endpoint: 'chat_completions'
    schema: 'llamacpp'
    instances: [{ url: 'not-a-url', api_key: 'k' }]
`
	_, err := Load(writeTempConfig(t, y))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "url")
}

func TestLogEffective_RedactsKeys(t *testing.T) {
	core, logs := observer.New(zap.InfoLevel)
	log := zap.New(core)

	cfg, err := Load(writeTempConfig(t, validYAML))
	assert.NoError(t, err)
	cfg.LogEffective(log)

	entries := logs.All()
	assert.Len(t, entries, 1)

	j, err := json.Marshal(entries[0].ContextMap())
	assert.NoError(t, err)
	rendered := string(j)
	assert.NotContains(t, rendered, "supersecretclientkey")
	assert.NotContains(t, rendered, "backendkey1234")
	assert.NotContains(t, rendered, "backendkey5678")
	assert.Contains(t, rendered, "tkey")
	assert.Contains(t, rendered, "1234")
	assert.Contains(t, rendered, "5678")
}

func TestRedact(t *testing.T) {
	assert.Equal(t, "", redact(""))
	assert.Equal(t, "****", redact("abc"))
	assert.Equal(t, "***abcd", redact("xyzabcd"))
}
