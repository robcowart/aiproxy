package schema

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// OpenAITranslator forwards requests/responses with minimal changes to an upstream OpenAI-compatible server.
type OpenAITranslator struct{}

// Name implements Translator.
func (*OpenAITranslator) Name() string { return "openai" }

// HealthPath returns the OpenAI health check path.
func (*OpenAITranslator) HealthPath() string { return "/v1/models" }

// ChatBackendRequest marshals the canonical request to /v1/chat/completions.
func (*OpenAITranslator) ChatBackendRequest(req *ChatRequest) (*BackendRequest, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal chat request: %w", err)
	}
	return &BackendRequest{
		Method: http.MethodPost,
		Path:   "/v1/chat/completions",
		Body:   body,
	}, nil
}

// ChatResponseFromBytes parses a non-stream backend response.
func (*OpenAITranslator) ChatResponseFromBytes(body []byte) (*ChatResponse, error) {
	var resp ChatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parse chat response: %w", err)
	}
	return &resp, nil
}

// NewChatStreamReader wraps the backend SSE response.
func (*OpenAITranslator) NewChatStreamReader(body io.ReadCloser) (StreamReader, error) {
	return &openAIStreamReader{body: body, sse: NewSSEScanner(body)}, nil
}

// EmbeddingsBackendRequest marshals the canonical request to /v1/embeddings.
func (*OpenAITranslator) EmbeddingsBackendRequest(req *EmbeddingsRequest) (*BackendRequest, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal embeddings request: %w", err)
	}
	return &BackendRequest{
		Method: http.MethodPost,
		Path:   "/v1/embeddings",
		Body:   body,
	}, nil
}

// EmbeddingsResponseFromBytes parses a backend embeddings response.
func (*OpenAITranslator) EmbeddingsResponseFromBytes(body []byte) (*EmbeddingsResponse, error) {
	var resp EmbeddingsResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parse embeddings response: %w", err)
	}
	return &resp, nil
}

// RerankBackendRequest marshals the canonical rerank request. OpenAI does not have an official rerank endpoint; we call
// /v1/rerank (as llama.cpp/jina/bge servers do).
func (*OpenAITranslator) RerankBackendRequest(req *RerankRequest) (*BackendRequest, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal rerank request: %w", err)
	}
	return &BackendRequest{
		Method: http.MethodPost,
		Path:   "/v1/rerank",
		Body:   body,
	}, nil
}

// RerankResponseFromBytes parses a backend rerank response.
func (*OpenAITranslator) RerankResponseFromBytes(body []byte) (*RerankResponse, error) {
	var resp RerankResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parse rerank response: %w", err)
	}
	return &resp, nil
}

// ModelsBackendRequest prepares a GET /v1/models. Shared by OpenAI-compatible servers (including llama.cpp).
func (*OpenAITranslator) ModelsBackendRequest() (*BackendRequest, error) {
	return &BackendRequest{Method: http.MethodGet, Path: "/v1/models"}, nil
}

// ModelsResponseFromBytes parses `{"data":[{id, object, created, owned_by, meta, ...}]}`. Unknown top-level fields
// on each entry (e.g., llama.cpp's `meta`) round-trip through ModelInfo.Meta.
func (*OpenAITranslator) ModelsResponseFromBytes(body []byte) ([]ModelInfo, error) {
	var wrapper struct {
		Data []ModelInfo `json:"data"`
	}
	if err := json.Unmarshal(body, &wrapper); err != nil {
		return nil, fmt.Errorf("parse models response: %w", err)
	}
	return wrapper.Data, nil
}

// openAIStreamReader re-emits backend SSE data frames verbatim (after parsing and re-marshaling to the canonical
// ChatStreamChunk shape, which preserves reasoning_content whether it appears on delta or message).
type openAIStreamReader struct {
	body io.ReadCloser
	sse  *SSEScanner
}

func (r *openAIStreamReader) Next() ([][]byte, bool, error) {
	for {
		ev, err := r.sse.Next()
		if err == io.EOF {
			return nil, true, nil
		}
		if err != nil {
			return nil, false, err
		}
		if ev == nil || len(ev.Data) == 0 {
			continue
		}
		data := bytes.TrimSpace(ev.Data)
		if bytes.Equal(data, []byte("[DONE]")) {
			return nil, true, nil
		}
		var chunk ChatStreamChunk
		if err := json.Unmarshal(data, &chunk); err != nil {
			return nil, false, fmt.Errorf("parse chat stream chunk: %w", err)
		}
		out, err := json.Marshal(chunk)
		if err != nil {
			return nil, false, fmt.Errorf("marshal chat stream chunk: %w", err)
		}
		return [][]byte{out}, false, nil
	}
}

func (r *openAIStreamReader) Close() error { return r.body.Close() }
