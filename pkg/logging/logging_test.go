package logging

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNew_Levels(t *testing.T) {
	for _, lvl := range []string{"", "debug", "info", "warn", "warning", "error", "INFO", "Debug"} {
		lg, err := New(lvl)
		assert.NoError(t, err, "level=%q", lvl)
		assert.NotNil(t, lg)
	}
}

func TestNew_InvalidLevel(t *testing.T) {
	_, err := New("bogus")
	assert.Error(t, err)
}

func TestNewNop(t *testing.T) {
	assert.NotNil(t, NewNop())
}
