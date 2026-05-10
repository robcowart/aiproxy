package frontend

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/robcowart/aiproxy/pkg/backend"
	"github.com/robcowart/aiproxy/pkg/config"
	"github.com/robcowart/aiproxy/pkg/logging"
	"github.com/robcowart/aiproxy/pkg/metrics"
	"github.com/robcowart/aiproxy/pkg/schema"
	"github.com/stretchr/testify/assert"
)

// counterValue returns the float64 value of a labeled CounterVec child, or 0 if absent.
func counterValue(t *testing.T, cv *prometheus.CounterVec, labels ...string) float64 {
	t.Helper()
	return testutil.ToFloat64(cv.WithLabelValues(labels...))
}

// buildTestServer wires a three-pool proxy (chat, embeddings, rerank) against three in-process fake llama.cpp backends
// and returns the handler plus a cleanup closure.
func buildTestServer(t *testing.T, chatH, embH, rrH http.HandlerFunc) (http.Handler, func()) {
	h, _, cleanup := buildTestServerWithMetrics(t, chatH, embH, rrH)
	return h, cleanup
}

func buildTestServerWithMetrics(t *testing.T, chatH, embH, rrH http.HandlerFunc) (http.Handler, *metrics.Metrics, func()) {
	t.Helper()
	chatSrv := httptest.NewServer(chatH)
	embSrv := httptest.NewServer(embH)
	rrSrv := httptest.NewServer(rrH)

	cfg := &config.Config{
		Server: config.ServerConfig{
			Host:   "127.0.0.1",
			Port:   8080,
			APIKey: "clientkey",
		},
		Log: config.LogConfig{Level: "info", Format: "json"},
		Pools: []config.PoolConfig{
			{
				Model: "chat-model", Endpoint: config.EndpointChatCompletions, Schema: config.SchemaLlamaCPP,
				SessionTimeout: 60, HealthCheckInterval: 30,
				Instances: []config.InstanceConfig{{URL: chatSrv.URL, APIKey: "beK"}},
			},
			{
				Model: "emb-model", Endpoint: config.EndpointEmbeddings, Schema: config.SchemaLlamaCPP,
				SessionTimeout: 60, HealthCheckInterval: 30,
				Instances: []config.InstanceConfig{{URL: embSrv.URL, APIKey: "beK"}},
			},
			{
				Model: "rerank-model", Endpoint: config.EndpointRerank, Schema: config.SchemaLlamaCPP,
				SessionTimeout: 60, HealthCheckInterval: 30,
				Instances: []config.InstanceConfig{{URL: rrSrv.URL, APIKey: "beK"}},
			},
		},
	}
	assert.NoError(t, cfg.Validate())

	reg, err := backend.NewRegistry(cfg, schema.NewRegistry())
	assert.NoError(t, err)

	m := metrics.New()
	fwd := backend.NewForwarder(logging.NewNop(), m)
	srv := NewFrontend(cfg, reg, fwd, logging.NewNop(), m)

	cleanup := func() {
		chatSrv.Close()
		embSrv.Close()
		rrSrv.Close()
	}
	return srv.Handler(), m, cleanup
}

func TestServer_ListModels(t *testing.T) {
	h, cleanup := buildTestServer(t,
		func(w http.ResponseWriter, _ *http.Request) {},
		func(w http.ResponseWriter, _ *http.Request) {},
		func(w http.ResponseWriter, _ *http.Request) {},
	)
	defer cleanup()

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var out struct {
		Data []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	assert.NoError(t, json.Unmarshal(w.Body.Bytes(), &out))
	var ids []string
	for _, m := range out.Data {
		ids = append(ids, m.ID)
		assert.Equal(t, "model", m.Object)
		assert.Equal(t, "llamacpp", m.OwnedBy)
	}
	assert.ElementsMatch(t, []string{"chat-model", "emb-model", "rerank-model"}, ids)
}

func TestServer_ListModels_EnrichesFromBackend(t *testing.T) {
	chatH := func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"object":"list",
				"data":[{
					"id":"chat-model",
					"object":"model",
					"created":1776774392,
					"owned_by":"llamacpp",
					"meta":{"n_ctx_train":1048576,"n_params":31577940288}
				}]
			}`))
			return
		}
	}
	h, cleanup := buildTestServer(t, chatH,
		func(w http.ResponseWriter, _ *http.Request) {},
		func(w http.ResponseWriter, _ *http.Request) {},
	)
	defer cleanup()

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	body := w.Body.String()
	assert.Contains(t, body, `"id":"chat-model"`)
	assert.Contains(t, body, `"created":1776774392`)
	assert.Contains(t, body, `"n_ctx_train":1048576`)
	assert.Contains(t, body, `"n_params":31577940288`)
	assert.Contains(t, body, `"id":"emb-model"`)
	assert.Contains(t, body, `"id":"rerank-model"`)
}

func TestServer_ListModels_EnrichesWhenBackendIDIsAliased(t *testing.T) {
	chatH := func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"object":"list",
				"data":[{
					"id":"chat-model-mxfp4_moe_bf16",
					"aliases":["chat-model"],
					"object":"model",
					"created":1777716931,
					"owned_by":"llamacpp",
					"meta":{"n_ctx_train":262144,"n_params":122111526912}
				}]
			}`))
			return
		}
	}
	h, cleanup := buildTestServer(t, chatH,
		func(w http.ResponseWriter, _ *http.Request) {},
		func(w http.ResponseWriter, _ *http.Request) {},
	)
	defer cleanup()

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	body := w.Body.String()
	assert.Contains(t, body, `"id":"chat-model"`)
	assert.NotContains(t, body, `"chat-model-mxfp4_moe_bf16"`)
	assert.NotContains(t, body, `"aliases"`)
	assert.Contains(t, body, `"created":1777716931`)
	assert.Contains(t, body, `"n_ctx_train":262144`)
	assert.Contains(t, body, `"n_params":122111526912`)
}

func TestServer_ListModels_EnrichesWhenBackendReportsSingleSuffixedModel(t *testing.T) {
	chatH := func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"object":"list",
				"data":[{
					"id":"chat-model-q8_0",
					"object":"model",
					"created":1777716931,
					"owned_by":"llamacpp",
					"meta":{"n_ctx_train":262144}
				}]
			}`))
			return
		}
	}
	h, cleanup := buildTestServer(t, chatH,
		func(w http.ResponseWriter, _ *http.Request) {},
		func(w http.ResponseWriter, _ *http.Request) {},
	)
	defer cleanup()

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	body := w.Body.String()
	assert.Contains(t, body, `"id":"chat-model"`)
	assert.NotContains(t, body, `"chat-model-q8_0"`)
	assert.Contains(t, body, `"n_ctx_train":262144`)
}

func TestServer_AuthRequired(t *testing.T) {
	h, cleanup := buildTestServer(t,
		func(w http.ResponseWriter, _ *http.Request) {},
		func(w http.ResponseWriter, _ *http.Request) {},
		func(w http.ResponseWriter, _ *http.Request) {},
	)
	defer cleanup()

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	assert.Equal(t, http.StatusUnauthorized, w.Code)
}

func TestServer_HealthAndMetricsBypassAuth(t *testing.T) {
	h, cleanup := buildTestServer(t,
		func(w http.ResponseWriter, _ *http.Request) {},
		func(w http.ResponseWriter, _ *http.Request) {},
		func(w http.ResponseWriter, _ *http.Request) {},
	)
	defer cleanup()

	for _, path := range []string{"/health", "/metrics"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code, "path=%s", path)
	}
}

func TestServer_ChatCompletion_NonStream(t *testing.T) {
	var seenAuth string
	chatH := func(w http.ResponseWriter, r *http.Request) {
		seenAuth = r.Header.Get("Authorization")
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		body, _ := io.ReadAll(r.Body)
		assert.Contains(t, string(body), `"model":"chat-model"`)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"1","object":"chat.completion","created":1,"model":"chat-model","choices":[{"index":0,"message":{"role":"assistant","content":"hi","reasoning_content":"thought"},"finish_reason":"stop"}]}`))
	}
	h, cleanup := buildTestServer(t, chatH, nil, nil)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"chat-model","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "Bearer beK", seenAuth)
	assert.Contains(t, w.Body.String(), `"reasoning_content":"thought"`)
}

func TestServer_ChatCompletion_Stream(t *testing.T) {
	chatH := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		for _, d := range []string{
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"chat-model","choices":[{"index":0,"delta":{"role":"assistant"}}]}`,
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"chat-model","choices":[{"index":0,"delta":{"reasoning_content":"thought"}}]}`,
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"chat-model","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		} {
			_, _ = w.Write([]byte("data: " + d + "\n\n"))
			f.Flush()
		}
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		f.Flush()
	}
	h, cleanup := buildTestServer(t, chatH, nil, nil)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"chat-model","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	body := w.Body.String()
	assert.Contains(t, body, `"role":"assistant"`)
	assert.Contains(t, body, `"reasoning_content":"thought"`)
	assert.Contains(t, body, `"content":"hi"`)
	assert.Contains(t, body, "data: [DONE]")
	assert.Equal(t, "text/event-stream", w.Header().Get("Content-Type"))
}

func TestServer_UnknownModel(t *testing.T) {
	h, cleanup := buildTestServer(t, nil, nil, nil)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"nope","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	assert.Equal(t, http.StatusNotFound, w.Code)
}

func TestServer_WrongEndpointForModel(t *testing.T) {
	h, cleanup := buildTestServer(t, nil, nil, nil)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"emb-model","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "wrong_endpoint")
}

func TestServer_Embeddings(t *testing.T) {
	embH := func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/embeddings", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","model":"emb-model","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}]}`))
	}
	h, cleanup := buildTestServer(t, nil, embH, nil)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"emb-model","input":"hello"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), `"embedding":[0.1,0.2,0.3]`)
}

func TestServer_Rerank(t *testing.T) {
	rrH := func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/rerank", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","model":"rerank-model","results":[{"index":0,"relevance_score":0.9}]}`))
	}
	h, cleanup := buildTestServer(t, nil, nil, rrH)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"rerank-model","query":"q","documents":["a","b"]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/rerank", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), `"relevance_score":0.9`)
}

func TestServer_ChatCompletion_NonStream_TokensRecorded(t *testing.T) {
	chatH := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"1","object":"chat.completion","created":1,"model":"chat-model",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}`))
	}
	h, m, cleanup := buildTestServerWithMetrics(t, chatH, nil, nil)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"chat-model","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, 7.0, counterValue(t, m.ClientPromptTokens, "chat-model", "chat_completions"))
	assert.Equal(t, 3.0, counterValue(t, m.ClientCompletionTokens, "chat-model", "chat_completions"))
}

func TestServer_Embeddings_TokensRecorded(t *testing.T) {
	embH := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","model":"emb-model","data":[{"object":"embedding","index":0,"embedding":[0.1]}],` +
			`"usage":{"prompt_tokens":9,"total_tokens":9}}`))
	}
	h, m, cleanup := buildTestServerWithMetrics(t, nil, embH, nil)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"emb-model","input":"hello"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, 9.0, counterValue(t, m.ClientPromptTokens, "emb-model", "embeddings"))
	assert.Equal(t, 0.0, counterValue(t, m.ClientCompletionTokens, "emb-model", "embeddings"))
}

func TestServer_Rerank_TokensRecorded(t *testing.T) {
	rrH := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","model":"rerank-model","results":[{"index":0,"relevance_score":0.9}]}`))
	}
	h, m, cleanup := buildTestServerWithMetrics(t, nil, nil, rrH)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"rerank-model","query":"q","documents":["a","b"]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/rerank", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, 0.0, counterValue(t, m.ClientPromptTokens, "rerank-model", "rerank"))
	assert.Equal(t, 0.0, counterValue(t, m.ClientCompletionTokens, "rerank-model", "rerank"))
}

// streamChatHandler returns an httptest handler that echoes two content chunks, a usage-only chunk, and [DONE]. It
// captures the last request body into capturedBody if non-nil.
func streamChatHandler(capturedBody *string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if capturedBody != nil {
			b, _ := io.ReadAll(r.Body)
			*capturedBody = string(b)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		for _, d := range []string{
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"chat-model","choices":[{"index":0,"delta":{"role":"assistant"}}]}`,
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"chat-model","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"chat-model","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}`,
		} {
			_, _ = w.Write([]byte("data: " + d + "\n\n"))
			f.Flush()
		}
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		f.Flush()
	}
}

func TestServer_ChatCompletion_Stream_DefaultIncludesUsageChunk(t *testing.T) {
	var backendBody string
	h, m, cleanup := buildTestServerWithMetrics(t, streamChatHandler(&backendBody), nil, nil)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"chat-model","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, backendBody, `"include_usage":true`)
	body := w.Body.String()
	assert.Contains(t, body, `"usage":{"prompt_tokens":7`)
	assert.Contains(t, body, "data: [DONE]")
	assert.Equal(t, 7.0, counterValue(t, m.ClientPromptTokens, "chat-model", "chat_completions"))
	assert.Equal(t, 3.0, counterValue(t, m.ClientCompletionTokens, "chat-model", "chat_completions"))
}

func TestServer_ChatCompletion_Stream_ExplicitIncludeUsageTrue(t *testing.T) {
	var backendBody string
	h, m, cleanup := buildTestServerWithMetrics(t, streamChatHandler(&backendBody), nil, nil)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"chat-model","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, backendBody, `"include_usage":true`)
	assert.Contains(t, w.Body.String(), `"usage":{"prompt_tokens":7`)
	assert.Equal(t, 7.0, counterValue(t, m.ClientPromptTokens, "chat-model", "chat_completions"))
	assert.Equal(t, 3.0, counterValue(t, m.ClientCompletionTokens, "chat-model", "chat_completions"))
}

func TestServer_ChatCompletion_Stream_ClientOptOutStripsUsageChunk(t *testing.T) {
	var backendBody string
	h, m, cleanup := buildTestServerWithMetrics(t, streamChatHandler(&backendBody), nil, nil)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"chat-model","stream":true,"stream_options":{"include_usage":false},"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, backendBody, `"include_usage":true`)
	body := w.Body.String()
	assert.NotContains(t, body, `"usage":{`)
	assert.Contains(t, body, "data: [DONE]")
	assert.Equal(t, 7.0, counterValue(t, m.ClientPromptTokens, "chat-model", "chat_completions"))
	assert.Equal(t, 3.0, counterValue(t, m.ClientCompletionTokens, "chat-model", "chat_completions"))
}

func TestSessionKey(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	r.Header.Set("X-Session-Id", "abc")
	k := sessionKey(r, "p")
	assert.True(t, strings.HasPrefix(k, "hdr:abc|p"))

	r2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	r2.RemoteAddr = "1.2.3.4:5555"
	k2 := sessionKey(r2, "p")
	assert.Equal(t, "conn:1.2.3.4:5555|p", k2)
}

// buildTestServerWithChatParams constructs a single-pool chat-only test server with the supplied static parameter
// overrides on the pool, plus a backend handler that captures the upstream request body for assertions.
func buildTestServerWithChatParams(t *testing.T, params map[string]any, chatH http.HandlerFunc) (http.Handler, func()) {
	t.Helper()
	chatSrv := httptest.NewServer(chatH)
	cfg := &config.Config{
		Server: config.ServerConfig{Host: "127.0.0.1", Port: 8080, APIKey: "clientkey"},
		Log:    config.LogConfig{Level: "info", Format: "json"},
		Pools: []config.PoolConfig{{
			Model: "chat-model", Endpoint: config.EndpointChatCompletions, Schema: config.SchemaLlamaCPP,
			SessionTimeout: 60, HealthCheckInterval: 30,
			Instances:  []config.InstanceConfig{{URL: chatSrv.URL, APIKey: "beK"}},
			Parameters: params,
		}},
	}
	assert.NoError(t, cfg.Validate())
	reg, err := backend.NewRegistry(cfg, schema.NewRegistry())
	assert.NoError(t, err)
	m := metrics.New()
	fwd := backend.NewForwarder(logging.NewNop(), m)
	srv := NewFrontend(cfg, reg, fwd, logging.NewNop(), m)
	return srv.Handler(), func() { chatSrv.Close() }
}

func TestServer_ChatCompletion_StaticParams_StreamFalseDrivesBranch(t *testing.T) {
	var seenBody string
	var seenAccept string
	chatH := func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		seenBody = string(body)
		seenAccept = r.Header.Get("Accept")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"1","object":"chat.completion","created":1,"model":"chat-model",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`))
	}
	h, cleanup := buildTestServerWithChatParams(t, map[string]any{
		"temperature": 0.2,
		"stream":      false,
		"top_k":       40,
	}, chatH)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"chat-model","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"),
		"override stream:false must drive non-streaming response branch")
	assert.NotContains(t, w.Body.String(), "data: ")
	assert.Contains(t, w.Body.String(), `"choices"`)

	assert.Contains(t, seenBody, `"temperature":0.2`)
	assert.Contains(t, seenBody, `"top_k":40`)
	// schema.ChatRequest.Stream is omitempty, so a false value drops the field entirely from the upstream body.
	// The branching evidence is the application/json Content-Type returned to the client above.
	assert.NotContains(t, seenBody, `"stream":true`)
	_ = seenAccept
}

func TestServer_ChatCompletion_StaticParams_StreamTrueOverridesClientFalse(t *testing.T) {
	var seenBody string
	chatH := func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		seenBody = string(body)
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		_, _ = w.Write([]byte(`data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"chat-model","choices":[{"index":0,"delta":{"content":"hi"}}]}` + "\n\n"))
		f.Flush()
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		f.Flush()
	}
	h, cleanup := buildTestServerWithChatParams(t, map[string]any{"stream": true}, chatH)
	defer cleanup()

	payload := bytes.NewBufferString(`{"model":"chat-model","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", payload)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer clientkey")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "text/event-stream", w.Header().Get("Content-Type"))
	assert.Contains(t, w.Body.String(), "data: [DONE]")
	assert.Contains(t, seenBody, `"stream":true`)
}
