package schema

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestOpenAITranslator_EmbeddingsAndRerank(t *testing.T) {
	tr := &OpenAITranslator{}
	embReq := &EmbeddingsRequest{Model: "m", Input: json.RawMessage(`"hello"`)}
	br, err := tr.EmbeddingsBackendRequest(embReq)
	assert.NoError(t, err)
	assert.Equal(t, "/v1/embeddings", br.Path)

	embResp, err := tr.EmbeddingsResponseFromBytes([]byte(`{"object":"list","model":"m","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}]}`))
	assert.NoError(t, err)
	assert.Len(t, embResp.Data, 1)

	rrReq := &RerankRequest{Model: "m", Query: "q", Documents: []string{"a", "b"}}
	br, err = tr.RerankBackendRequest(rrReq)
	assert.NoError(t, err)
	assert.Equal(t, "/v1/rerank", br.Path)

	rrResp, err := tr.RerankResponseFromBytes([]byte(`{"object":"list","model":"m","results":[{"index":0,"relevance_score":0.9}]}`))
	assert.NoError(t, err)
	assert.Equal(t, float64(0.9), rrResp.Results[0].RelevanceScore)
}
