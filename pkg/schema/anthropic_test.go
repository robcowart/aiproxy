package schema

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAnthropicTranslator_ChatRequest(t *testing.T) {
	tr := &AnthropicTranslator{}
	mx := 128
	req := &ChatRequest{
		Model:     "claude-3",
		MaxTokens: &mx,
		Messages: []Message{
			{Role: "system", Content: json.RawMessage(`"be terse"`)},
			{Role: "user", Content: json.RawMessage(`"hi"`)},
		},
	}
	br, err := tr.ChatBackendRequest(req)
	assert.NoError(t, err)
	assert.Equal(t, "/v1/messages", br.Path)
	assert.Equal(t, "2023-06-01", br.Headers.Get("anthropic-version"))
	body := string(br.Body)
	assert.Contains(t, body, `"system":"be terse"`)
	assert.Contains(t, body, `"max_tokens":128`)
}

func TestAnthropicTranslator_ChatResponseIncludesReasoning(t *testing.T) {
	tr := &AnthropicTranslator{}
	body := []byte(`{"id":"m1","model":"claude","role":"assistant","content":[{"type":"thinking","thinking":"thought"},{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":5}}`)
	resp, err := tr.ChatResponseFromBytes(body)
	assert.NoError(t, err)
	assert.Equal(t, "thought", resp.Choices[0].Message.ReasoningContent)
	var content string
	assert.NoError(t, json.Unmarshal(resp.Choices[0].Message.Content, &content))
	assert.Equal(t, "hello", content)
	assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	assert.Equal(t, 8, resp.Usage.TotalTokens)
}

func TestAnthropicTranslator_Stream(t *testing.T) {
	tr := &AnthropicTranslator{}
	sse := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"claude\",\"usage\":{\"input_tokens\":11}}}\n\n" +
		"event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n\n" +
		"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"ponder\"}}\n\n" +
		"event: content_block_start\ndata: {\"index\":1,\"content_block\":{\"type\":\"text\"}}\n\n" +
		"event: content_block_delta\ndata: {\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
		"event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n" +
		"event: message_stop\ndata: {}\n\n"
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
	joined := ""
	for _, c := range all {
		joined += string(c) + "|"
	}
	assert.Contains(t, joined, `"role":"assistant"`)
	assert.Contains(t, joined, `"reasoning_content":"ponder"`)
	assert.Contains(t, joined, `"content":"hi"`)
	assert.Contains(t, joined, `"finish_reason":"stop"`)

	var last ChatStreamChunk
	assert.NoError(t, json.Unmarshal(all[len(all)-1], &last))
	assert.Empty(t, last.Choices)
	assert.NotNil(t, last.Usage)
	assert.Equal(t, 11, last.Usage.PromptTokens)
	assert.Equal(t, 4, last.Usage.CompletionTokens)
	assert.Equal(t, 15, last.Usage.TotalTokens)
}

func TestAnthropicTranslator_UnsupportedEndpoints(t *testing.T) {
	tr := &AnthropicTranslator{}
	_, err := tr.EmbeddingsBackendRequest(&EmbeddingsRequest{})
	assert.True(t, errors.Is(err, ErrUnsupportedEndpoint))
	_, err = tr.RerankBackendRequest(&RerankRequest{})
	assert.True(t, errors.Is(err, ErrUnsupportedEndpoint))
}

func TestAnthropicTranslator_Models(t *testing.T) {
	tr := &AnthropicTranslator{}
	breq, err := tr.ModelsBackendRequest()
	assert.NoError(t, err)
	assert.Equal(t, "/v1/models", breq.Path)

	body := []byte(`{"data":[{"id":"claude-3-5-sonnet-20241022","type":"model","display_name":"Claude 3.5 Sonnet","created_at":"2024-10-22T00:00:00Z"}]}`)
	models, err := tr.ModelsResponseFromBytes(body)
	assert.NoError(t, err)
	assert.Len(t, models, 1)
	assert.Equal(t, "claude-3-5-sonnet-20241022", models[0].ID)
	assert.Equal(t, "anthropic", models[0].OwnedBy)
	assert.Equal(t, int64(1729555200), models[0].Created)
	assert.Contains(t, string(models[0].Meta), `"display_name":"Claude 3.5 Sonnet"`)
}
