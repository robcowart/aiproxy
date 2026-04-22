package schema

import "io"

// LlamaCPPTranslator is an OpenAI-compatible translator tuned for llama.cpp servers. The wire format is identical to
// OpenAI; the only differences are the health-check path and the fact that llama.cpp places chain-of-thought output in
// `message.reasoning_content` / `delta.reasoning_content`, which our canonical types already carry through.
type LlamaCPPTranslator struct{ OpenAITranslator }

// Name implements Translator.
func (*LlamaCPPTranslator) Name() string { return "llamacpp" }

// HealthPath returns the llama.cpp /health endpoint.
func (*LlamaCPPTranslator) HealthPath() string { return "/health" }

// NewChatStreamReader is inherited from OpenAITranslator but re-declared here to surface the method on the embedded struct
// via the Translator interface.
func (t *LlamaCPPTranslator) NewChatStreamReader(body io.ReadCloser) (StreamReader, error) {
	return t.OpenAITranslator.NewChatStreamReader(body)
}
