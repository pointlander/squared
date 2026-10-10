// Copyright 2026 The squared Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command squared trains a byte model and continues a prompt.
//
// Each distinct byte is a symbol. Column t of A holds that symbol's embedding
// in a private block for position t and again in a shared band, then the
// column is scaled to unit length. The private blocks keep the order. The
// shared band lets the solved X mix the window. Gradient descent solves
// X^2 A = A. The solved X is the feedforward network's embedding. While
// training, entries of X are dropped, and the network takes its own gradient
// step. -n counts bytes. A second network reads the columns and skips the
// solve; its held-out loss is printed as cols.
//
//	squared -prompt "To be, or not to be" -n 80
package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	file := flag.String("file", "pg100.txt", "corpus file")
	promptFlag := flag.String("prompt", "To be, or not to be", "prompt to continue")
	n := flag.Int("n", 80, "bytes to generate after the prompt")
	seed := flag.Int64("seed", 1, "random seed")
	steps := flag.Int("steps", 400, "Adam steps")
	flag.Parse()

	if err := runX2(*file, *promptFlag, *n, *seed, *steps); err != nil {
		fmt.Fprintf(os.Stderr, "squared: %s\n", err)
		os.Exit(1)
	}
}
