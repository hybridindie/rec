package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func launchCommand(p Paths, m Manifest, runtimeDir, model string, port int) (string, []string, []string) {
	args := []string{
		"serve", m.Model, "--model-dir", filepath.Dir(model), "--manifest", model,
		"--port", strconv.Itoa(port), "--context-limit", strconv.Itoa(m.Context),
		"--api-key-file", filepath.Join(p.Config, "api-key"),
	}
	// REC_ENGINE_ARGS passes extra engine flags through, e.g. "--metrics /path.jsonl".
	args = append(args, strings.Fields(os.Getenv("REC_ENGINE_ARGS"))...)
	return filepath.Join(runtimeDir, runtimeServer), args, []string{"HF_HUB_OFFLINE=1", "TOKENIZERS_PARALLELISM=false"}
}

func checkAppleChip(name string) error {
	for _, chip := range []string{"Apple M4 Pro", "Apple M5 Pro", "Apple M5 Max"} {
		if name == chip {
			return nil
		}
	}
	return fmt.Errorf("Flint requires an Apple M4 Pro, M5 Pro or M5 Max; detected %s", name)
}
