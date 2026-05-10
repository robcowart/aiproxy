package logging

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNew_Levels(t *testing.T) {
	for _, lvl := range []string{"", "debug", "info", "warn", "warning", "error", "INFO", "Debug"} {
		lg, err := New(lvl, "")
		assert.NoError(t, err, "level=%q", lvl)
		assert.NotNil(t, lg)
	}
}

func TestNew_Formats(t *testing.T) {
	for _, fmt := range []string{"", "json", "JSON", "console", "Console"} {
		lg, err := New("info", fmt)
		assert.NoError(t, err, "format=%q", fmt)
		assert.NotNil(t, lg)
	}
}

func TestNew_InvalidLevel(t *testing.T) {
	_, err := New("bogus", "json")
	assert.Error(t, err)
}

func TestNew_InvalidFormat(t *testing.T) {
	_, err := New("info", "xml")
	assert.Error(t, err)
}

func TestNewNop(t *testing.T) {
	assert.NotNil(t, NewNop())
}
