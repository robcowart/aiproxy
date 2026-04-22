package schema

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestOllamaTranslator_NameAndHealth(t *testing.T) {
	tr := &OllamaTranslator{}
	assert.Equal(t, "ollama", tr.Name())
	assert.Equal(t, "/", tr.HealthPath())
}

func TestOllamaTranslator_ChatBackendRequest_OptionsAndFormat(t *testing.T) {
	tr := &OllamaTranslator{}
	temp := 0.7
	topP := 0.9
	maxTok := 256
	seed := int64(42)
	req := &ChatRequest{
		Model:           "llama3.2",
		Stream:          true,
		Temperature:     &temp,
		TopP:            &topP,
		MaxTokens:       &maxTok,
		Seed:            &seed,
		Stop:            json.RawMessage(`["<|eot|>", "</s>"]`),
		ReasoningEffort: "medium",
		ResponseFormat:  json.RawMessage(`{"type":"json_object"}`),
		Messages: []Message{
			{Role: "system", Content: json.RawMessage(`"be brief"`)},
			{Role: "user", Content: json.RawMessage(`"hi"`)},
		},
	}

	br, err := tr.ChatBackendRequest(req)
	assert.NoError(t, err)
	assert.Equal(t, "POST", br.Method)
	assert.Equal(t, "/api/chat", br.Path)

	var body ollamaChatRequest
	assert.NoError(t, json.Unmarshal(br.Body, &body))
	assert.Equal(t, "llama3.2", body.Model)
	assert.True(t, body.Stream)
	assert.NotNil(t, body.Options)
	assert.Equal(t, 0.7, *body.Options.Temperature)
	assert.Equal(t, 256, *body.Options.NumPredict)
	assert.Equal(t, []string{"<|eot|>", "</s>"}, body.Options.Stop)
	assert.Equal(t, int64(42), *body.Options.Seed)
	assert.NotNil(t, body.Think)
	assert.True(t, *body.Think)
	assert.JSONEq(t, `"json"`, string(body.Format))
	assert.Len(t, body.Messages, 2)
	assert.Equal(t, "system", body.Messages[0].Role)
	assert.Equal(t, "be brief", body.Messages[0].Content)
	assert.Equal(t, "user", body.Messages[1].Role)
	assert.Equal(t, "hi", body.Messages[1].Content)
}

func TestOllamaTranslator_ChatBackendRequest_JSONSchemaFormat(t *testing.T) {
	tr := &OllamaTranslator{}
	req := &ChatRequest{
		Model:          "llama3.2",
		Messages:       []Message{{Role: "user", Content: json.RawMessage(`"ping"`)}},
		ResponseFormat: json.RawMessage(`{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"x":{"type":"number"}}}}}`),
	}
	br, err := tr.ChatBackendRequest(req)
	assert.NoError(t, err)

	var body ollamaChatRequest
	assert.NoError(t, json.Unmarshal(br.Body, &body))
	assert.JSONEq(t, `{"type":"object","properties":{"x":{"type":"number"}}}`, string(body.Format))
}

func TestOllamaTranslator_ChatBackendRequest_OmitsOptionsWhenEmpty(t *testing.T) {
	tr := &OllamaTranslator{}
	req := &ChatRequest{
		Model:    "llama3.2",
		Messages: []Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	}
	br, err := tr.ChatBackendRequest(req)
	assert.NoError(t, err)

	var raw map[string]json.RawMessage
	assert.NoError(t, json.Unmarshal(br.Body, &raw))
	_, hasOptions := raw["options"]
	assert.False(t, hasOptions)
	_, hasThink := raw["think"]
	assert.False(t, hasThink)
	_, hasFormat := raw["format"]
	assert.False(t, hasFormat)
}

func TestOllamaTranslator_ChatResponseFromBytes(t *testing.T) {
	tr := &OllamaTranslator{}
	body := []byte(`{
		"model":"llama3.2",
		"created_at":"2025-01-02T03:04:05.678901234Z",
		"message":{"role":"assistant","content":"blue","thinking":"the sky is blue"},
		"done":true,
		"done_reason":"stop",
		"prompt_eval_count":12,
		"eval_count":34
	}`)
	resp, err := tr.ChatResponseFromBytes(body)
	assert.NoError(t, err)
	assert.Equal(t, "llama3.2", resp.Model)
	assert.Equal(t, "chat.completion", resp.Object)
	assert.Len(t, resp.Choices, 1)
	assert.Equal(t, "assistant", resp.Choices[0].Message.Role)
	assert.Equal(t, "the sky is blue", resp.Choices[0].Message.ReasoningContent)
	assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	var content string
	assert.NoError(t, json.Unmarshal(resp.Choices[0].Message.Content, &content))
	assert.Equal(t, "blue", content)
	assert.NotNil(t, resp.Usage)
	assert.Equal(t, 12, resp.Usage.PromptTokens)
	assert.Equal(t, 34, resp.Usage.CompletionTokens)
	assert.Equal(t, 46, resp.Usage.TotalTokens)
}

func TestOllamaTranslator_ChatResponseFromBytes_ToolCalls(t *testing.T) {
	tr := &OllamaTranslator{}
	body := []byte(`{
		"model":"llama3.2",
		"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Paris"}}}]},
		"done":true,
		"done_reason":"stop"
	}`)
	resp, err := tr.ChatResponseFromBytes(body)
	assert.NoError(t, err)
	assert.NotNil(t, resp.Choices[0].Message.ToolCalls)
	var tcs []struct {
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}
	assert.NoError(t, json.Unmarshal(resp.Choices[0].Message.ToolCalls, &tcs))
	assert.Len(t, tcs, 1)
	assert.Equal(t, "function", tcs[0].Type)
	assert.Equal(t, "get_weather", tcs[0].Function.Name)
	assert.JSONEq(t, `{"city":"Paris"}`, tcs[0].Function.Arguments)
}

func TestOllamaTranslator_StreamNDJSON(t *testing.T) {
	tr := &OllamaTranslator{}
	nd := `{"model":"llama3.2","created_at":"2025-01-02T03:04:05Z","message":{"role":"assistant","content":"hel"},"done":false}
{"model":"llama3.2","created_at":"2025-01-02T03:04:05Z","message":{"role":"assistant","content":"lo"},"done":false}
{"model":"llama3.2","created_at":"2025-01-02T03:04:05Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":7}
`
	r, err := tr.NewChatStreamReader(nopCloser{strings.NewReader(nd)})
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
	// Expect: role chunk, "hel" delta, "lo" delta, finish chunk, usage-only chunk.
	assert.Len(t, all, 5)

	var role ChatStreamChunk
	assert.NoError(t, json.Unmarshal(all[0], &role))
	assert.Equal(t, "assistant", role.Choices[0].Delta.Role)

	var d1, d2 ChatStreamChunk
	assert.NoError(t, json.Unmarshal(all[1], &d1))
	assert.NoError(t, json.Unmarshal(all[2], &d2))
	assert.Equal(t, "hel", d1.Choices[0].Delta.Content)
	assert.Equal(t, "lo", d2.Choices[0].Delta.Content)

	var finish ChatStreamChunk
	assert.NoError(t, json.Unmarshal(all[3], &finish))
	assert.Equal(t, "stop", finish.Choices[0].FinishReason)

	var usage ChatStreamChunk
	assert.NoError(t, json.Unmarshal(all[4], &usage))
	assert.Empty(t, usage.Choices)
	assert.NotNil(t, usage.Usage)
	assert.Equal(t, 5, usage.Usage.PromptTokens)
	assert.Equal(t, 7, usage.Usage.CompletionTokens)
	assert.Equal(t, 12, usage.Usage.TotalTokens)
}

func TestOllamaTranslator_Embeddings(t *testing.T) {
	tr := &OllamaTranslator{}
	req := &EmbeddingsRequest{Model: "nomic-embed-text", Input: json.RawMessage(`["alpha","beta"]`)}
	br, err := tr.EmbeddingsBackendRequest(req)
	assert.NoError(t, err)
	assert.Equal(t, "/api/embed", br.Path)
	assert.Equal(t, "POST", br.Method)
	assert.Contains(t, string(br.Body), `"model":"nomic-embed-text"`)
	assert.Contains(t, string(br.Body), `"input":["alpha","beta"]`)

	body := []byte(`{"model":"nomic-embed-text","embeddings":[[0.1,0.2],[0.3,0.4]],"prompt_eval_count":8}`)
	resp, err := tr.EmbeddingsResponseFromBytes(body)
	assert.NoError(t, err)
	assert.Equal(t, "list", resp.Object)
	assert.Equal(t, "nomic-embed-text", resp.Model)
	assert.Len(t, resp.Data, 2)
	assert.Equal(t, 0, resp.Data[0].Index)
	assert.JSONEq(t, `[0.1,0.2]`, string(resp.Data[0].Embedding))
	assert.Equal(t, 1, resp.Data[1].Index)
	assert.NotNil(t, resp.Usage)
	assert.Equal(t, 8, resp.Usage.PromptTokens)
	assert.Equal(t, 8, resp.Usage.TotalTokens)
}

func TestOllamaTranslator_RerankUnsupported(t *testing.T) {
	tr := &OllamaTranslator{}
	_, err := tr.RerankBackendRequest(&RerankRequest{})
	assert.ErrorIs(t, err, ErrUnsupportedEndpoint)
	_, err = tr.RerankResponseFromBytes(nil)
	assert.ErrorIs(t, err, ErrUnsupportedEndpoint)
}

func TestOllamaTranslator_Models(t *testing.T) {
	tr := &OllamaTranslator{}
	breq, err := tr.ModelsBackendRequest()
	assert.NoError(t, err)
	assert.Equal(t, "/api/tags", breq.Path)
	assert.Equal(t, "GET", breq.Method)

	body := []byte(`{
		"models":[
			{"name":"llama3.2:latest","model":"llama3.2:latest","modified_at":"2024-12-01T10:20:30Z","size":4000000000,"digest":"abc","details":{"family":"llama"}},
			{"name":"qwen2.5:7b","model":"qwen2.5:7b","modified_at":"2024-11-01T00:00:00Z"}
		]
	}`)
	models, err := tr.ModelsResponseFromBytes(body)
	assert.NoError(t, err)
	assert.Len(t, models, 2)
	assert.Equal(t, "llama3.2:latest", models[0].ID)
	assert.Equal(t, "ollama", models[0].OwnedBy)
	assert.NotZero(t, models[0].Created)
	assert.Contains(t, string(models[0].Meta), `"digest":"abc"`)
	assert.Contains(t, string(models[0].Meta), `"family":"llama"`)
	assert.Equal(t, "qwen2.5:7b", models[1].ID)
}
