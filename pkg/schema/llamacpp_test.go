package schema

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLlamaCPPTranslator_ChatRequestResponse(t *testing.T) {
	tr := &LlamaCPPTranslator{}
	req := &ChatRequest{
		Model:    "qwen3",
		Messages: []Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	}
	br, err := tr.ChatBackendRequest(req)
	assert.NoError(t, err)
	assert.Equal(t, "/v1/chat/completions", br.Path)
	assert.Equal(t, "POST", br.Method)
	assert.Contains(t, string(br.Body), `"model":"qwen3"`)

	body := []byte(`{"id":"1","object":"chat.completion","created":1,"model":"qwen3","choices":[{"index":0,"message":{"role":"assistant","content":"hello","reasoning_content":"think"},"finish_reason":"stop"}]}`)
	resp, err := tr.ChatResponseFromBytes(body)
	assert.NoError(t, err)
	assert.Equal(t, "think", resp.Choices[0].Message.ReasoningContent)
}

func TestLlamaCPPTranslator_StreamPreservesReasoning(t *testing.T) {
	tr := &LlamaCPPTranslator{}
	sse := "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"ponder\"}}]}\n\n" +
		"data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: [DONE]\n\n"
	r, err := tr.NewChatStreamReader(nopCloser{strings.NewReader(sse)})
	assert.NoError(t, err)
	var all [][]byte
	for {
		chunks, done, err := r.Next()
		assert.NoError(t, err)
		all = append(all, chunks...)
		if done {
			break
		}
	}
	assert.Len(t, all, 3)
	assert.Contains(t, string(all[1]), `"reasoning_content":"ponder"`)
	assert.Contains(t, string(all[2]), `"content":"hi"`)
}

func TestLlamaCPPTranslator_HealthPath(t *testing.T) {
	tr := &LlamaCPPTranslator{}
	assert.Equal(t, "/health", tr.HealthPath())
}

func TestLlamaCPPTranslator_PreservesUsageDetailsAndTimings(t *testing.T) {
	tr := &LlamaCPPTranslator{}
	body := []byte(`{
		"id":"chatcmpl-x","object":"chat.completion","created":1,"model":"m",
		"system_fingerprint":"b8808-408225bb1",
		"choices":[{"index":0,"message":{"role":"assistant","content":"blue"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":36,"completion_tokens":99,"total_tokens":135,"prompt_tokens_details":{"cached_tokens":12}},
		"timings":{"cache_n":0,"prompt_n":36,"prompt_ms":57.742,"predicted_n":99,"predicted_ms":673.677,"predicted_per_second":146.95}
	}`)

	resp, err := tr.ChatResponseFromBytes(body)
	assert.NoError(t, err)
	assert.NotNil(t, resp.Usage)
	assert.Equal(t, 36, resp.Usage.PromptTokens)
	assert.JSONEq(t, `{"cached_tokens":12}`, string(resp.Usage.PromptTokensDetails))
	assert.NotEmpty(t, resp.Timings)

	out, err := json.Marshal(resp)
	assert.NoError(t, err)
	assert.Contains(t, string(out), `"prompt_tokens_details":{"cached_tokens":12}`)
	assert.Contains(t, string(out), `"timings":`)
	assert.Contains(t, string(out), `"predicted_per_second":146.95`)
}

func TestLlamaCPPTranslator_Models(t *testing.T) {
	tr := &LlamaCPPTranslator{}
	breq, err := tr.ModelsBackendRequest()
	assert.NoError(t, err)
	assert.Equal(t, "/v1/models", breq.Path)
	assert.Equal(t, "GET", breq.Method)

	body := []byte(`{
		"object":"list",
		"data":[{
			"id":"nemotron-cascade-2-30b",
			"object":"model",
			"created":1776774392,
			"owned_by":"llamacpp",
			"meta":{"vocab_type":2,"n_vocab":131072,"n_ctx_train":1048576,"n_embd":2688,"n_params":31577940288,"size":17972232960}
		}]
	}`)
	models, err := tr.ModelsResponseFromBytes(body)
	assert.NoError(t, err)
	assert.Len(t, models, 1)
	assert.Equal(t, "nemotron-cascade-2-30b", models[0].ID)
	assert.Equal(t, int64(1776774392), models[0].Created)
	assert.Equal(t, "llamacpp", models[0].OwnedBy)
	assert.Contains(t, string(models[0].Meta), `"n_ctx_train":1048576`)

	out, err := json.Marshal(models[0])
	assert.NoError(t, err)
	assert.Contains(t, string(out), `"created":1776774392`)
	assert.Contains(t, string(out), `"meta":{`)
}

func TestLlamaCPPTranslator_StreamPreservesUsageOnlyChunk(t *testing.T) {
	tr := &LlamaCPPTranslator{}
	sse := "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\"," +
		"\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\"," +
		"\"choices\":[]," +
		"\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7,\"total_tokens\":12}}\n\n" +
		"data: [DONE]\n\n"
	r, err := tr.NewChatStreamReader(nopCloser{strings.NewReader(sse)})
	assert.NoError(t, err)
	var all [][]byte
	for {
		chunks, done, err := r.Next()
		assert.NoError(t, err)
		all = append(all, chunks...)
		if done {
			break
		}
	}
	assert.Len(t, all, 2)
	var last ChatStreamChunk
	assert.NoError(t, json.Unmarshal(all[1], &last))
	assert.Empty(t, last.Choices)
	assert.NotNil(t, last.Usage)
	assert.Equal(t, 5, last.Usage.PromptTokens)
	assert.Equal(t, 7, last.Usage.CompletionTokens)
	assert.Equal(t, 12, last.Usage.TotalTokens)
}

func TestLlamaCPPTranslator_StreamPreservesUsageDetailsAndTimings(t *testing.T) {
	tr := &LlamaCPPTranslator{}
	sse := "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\"," +
		"\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]," +
		"\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7,\"total_tokens\":12,\"prompt_tokens_details\":{\"cached_tokens\":3}}," +
		"\"timings\":{\"predicted_per_second\":123.4}}\n\n" +
		"data: [DONE]\n\n"
	r, err := tr.NewChatStreamReader(nopCloser{strings.NewReader(sse)})
	assert.NoError(t, err)
	var all [][]byte
	for {
		chunks, done, err := r.Next()
		assert.NoError(t, err)
		all = append(all, chunks...)
		if done {
			break
		}
	}
	assert.Len(t, all, 1)
	assert.Contains(t, string(all[0]), `"prompt_tokens_details":{"cached_tokens":3}`)
	assert.Contains(t, string(all[0]), `"timings":{"predicted_per_second":123.4}`)
}
