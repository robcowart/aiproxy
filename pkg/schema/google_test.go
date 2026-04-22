package schema

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGoogleTranslator_ChatRequestRouting(t *testing.T) {
	tr := &GoogleTranslator{}
	req := &ChatRequest{
		Model:    "gemini-2.0",
		Messages: []Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	}
	br, err := tr.ChatBackendRequest(req)
	assert.NoError(t, err)
	assert.Contains(t, br.Path, ":generateContent")

	req.Stream = true
	br, err = tr.ChatBackendRequest(req)
	assert.NoError(t, err)
	assert.Contains(t, br.Path, ":streamGenerateContent")
	assert.Contains(t, br.Path, "alt=sse")
}

func TestGoogleTranslator_ChatResponseThought(t *testing.T) {
	tr := &GoogleTranslator{}
	body := []byte(`{"candidates":[{"content":{"parts":[{"text":"ponder","thought":true},{"text":"hi"}]},"finishReason":"STOP","index":0}],"modelVersion":"gemini-2.0","usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}}`)
	resp, err := tr.ChatResponseFromBytes(body)
	assert.NoError(t, err)
	assert.Equal(t, "ponder", resp.Choices[0].Message.ReasoningContent)
	var content string
	assert.NoError(t, json.Unmarshal(resp.Choices[0].Message.Content, &content))
	assert.Equal(t, "hi", content)
	assert.Equal(t, "stop", resp.Choices[0].FinishReason)
}

func TestGoogleTranslator_Embeddings(t *testing.T) {
	tr := &GoogleTranslator{}
	req := &EmbeddingsRequest{Model: "text-embedding", Input: json.RawMessage(`["a","b"]`)}
	br, err := tr.EmbeddingsBackendRequest(req)
	assert.NoError(t, err)
	assert.Contains(t, br.Path, ":batchEmbedContents")

	resp, err := tr.EmbeddingsResponseFromBytes([]byte(`{"embeddings":[{"values":[0.1,0.2]},{"values":[0.3,0.4]}]}`))
	assert.NoError(t, err)
	assert.Len(t, resp.Data, 2)
}

func TestGoogleTranslator_RerankUnsupported(t *testing.T) {
	tr := &GoogleTranslator{}
	_, err := tr.RerankBackendRequest(&RerankRequest{})
	assert.True(t, errors.Is(err, ErrUnsupportedEndpoint))
}

func TestGoogleTranslator_Stream(t *testing.T) {
	tr := &GoogleTranslator{}
	sse := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ponder\",\"thought\":true}]},\"index\":0}],\"modelVersion\":\"gemini-2.0\"}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]},\"index\":0}],\"modelVersion\":\"gemini-2.0\"}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\",\"index\":0}],\"modelVersion\":\"gemini-2.0\",\"usageMetadata\":{\"promptTokenCount\":6,\"candidatesTokenCount\":4,\"totalTokenCount\":10}}\n\n"
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
	assert.Equal(t, 6, last.Usage.PromptTokens)
	assert.Equal(t, 4, last.Usage.CompletionTokens)
	assert.Equal(t, 10, last.Usage.TotalTokens)
}

func TestGoogleTranslator_Models(t *testing.T) {
	tr := &GoogleTranslator{}
	breq, err := tr.ModelsBackendRequest()
	assert.NoError(t, err)
	assert.Equal(t, "/v1beta/models", breq.Path)

	body := []byte(`{"models":[{"name":"models/gemini-1.5-pro","version":"001","displayName":"Gemini 1.5 Pro","inputTokenLimit":1048576,"supportedGenerationMethods":["generateContent"]}]}`)
	models, err := tr.ModelsResponseFromBytes(body)
	assert.NoError(t, err)
	assert.Len(t, models, 1)
	assert.Equal(t, "gemini-1.5-pro", models[0].ID)
	assert.Equal(t, "google", models[0].OwnedBy)
	assert.Contains(t, string(models[0].Meta), `"name":"models/gemini-1.5-pro"`)
	assert.Contains(t, string(models[0].Meta), `"inputTokenLimit":1048576`)
}
