package schema

import (
	"errors"
	"fmt"
	"io"
	"net/http"
)

// BackendRequest is a translator-prepared HTTP request to a backend instance. Path is appended to the backend
// instance's base URL; headers (other than Authorization, which is injected by the forwarder) may be set here.
type BackendRequest struct {
	Method  string
	Path    string
	Body    []byte
	Headers http.Header
}

// StreamReader streams canonical OpenAI ChatStreamChunk JSON payloads from a provider-specific backend stream. Next
// returns zero or more chunk payloads (without any "data: " SSE prefix) along with a done flag indicating the stream
// has been fully consumed.
type StreamReader interface {
	Next() (chunks [][]byte, done bool, err error)
	Close() error
}

// Translator converts canonical OpenAI request/response shapes to a specific backend provider's native API and vice
// versa.
type Translator interface {
	Name() string
	HealthPath() string

	ChatBackendRequest(req *ChatRequest) (*BackendRequest, error)
	ChatResponseFromBytes(body []byte) (*ChatResponse, error)
	NewChatStreamReader(body io.ReadCloser) (StreamReader, error)

	EmbeddingsBackendRequest(req *EmbeddingsRequest) (*BackendRequest, error)
	EmbeddingsResponseFromBytes(body []byte) (*EmbeddingsResponse, error)

	RerankBackendRequest(req *RerankRequest) (*BackendRequest, error)
	RerankResponseFromBytes(body []byte) (*RerankResponse, error)

	// ModelsBackendRequest prepares a GET for the backend's models listing endpoint.
	ModelsBackendRequest() (*BackendRequest, error)
	// ModelsResponseFromBytes parses the backend response into canonical ModelInfo entries.
	ModelsResponseFromBytes(body []byte) ([]ModelInfo, error)
}

// ErrUnsupportedEndpoint is returned when a translator does not implement a given endpoint (e.g., Anthropic has no
// rerank API).
var ErrUnsupportedEndpoint = errors.New("endpoint not supported by backend provider")

// Registry is the lookup table of translators by config schema name.
type Registry struct {
	byName map[string]Translator
}

// NewRegistry returns the default registry populated with all built-in translators (llamacpp, openai, anthropic,
// google, ollama).
func NewRegistry() *Registry {
	r := &Registry{byName: make(map[string]Translator, 5)}
	r.Register(&LlamaCPPTranslator{})
	r.Register(&OpenAITranslator{})
	r.Register(&AnthropicTranslator{})
	r.Register(&GoogleTranslator{})
	r.Register(&OllamaTranslator{})
	return r
}

// Register adds or replaces a translator by its Name().
func (r *Registry) Register(t Translator) { r.byName[t.Name()] = t }

// Get returns the translator registered under name, or an error if unknown.
func (r *Registry) Get(name string) (Translator, error) {
	t, ok := r.byName[name]
	if !ok {
		return nil, fmt.Errorf("unknown schema %q", name)
	}
	return t, nil
}

// Names returns the sorted list of registered translator names.
func (r *Registry) Names() []string {
	names := make([]string, 0, len(r.byName))
	for n := range r.byName {
		names = append(names, n)
	}
	return names
}
