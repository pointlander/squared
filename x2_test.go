// Copyright 2026 The squared Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestX2ContextResidual(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	m := newX2Model(defaultX2Config(), rng)
	sc := newX2Scratch(m.cfg)
	text := []byte("To be, or not to be, that is the question")
	sum := 0.0
	for pos := 0; pos < len(text); pos++ {
		ctx := x2Context(text, pos, m.cfg.window, m.pad())
		m.buildA(ctx, sc.a)
		normalizeCols(sc.a, sc.ahat, sc.norms, m.cfg.dim, m.cfg.window)
		sum += sc.solve(sc.ahat, m.cfg.innerEta, m.cfg.lambda, m.cfg.x0)
	}
	mean := sum / float64(len(text))
	if mean > 0.08 {
		t.Fatalf("mean relative residual %g", mean)
	}
}

func TestX2SharedBandMixes(t *testing.T) {
	cfg := x2Config{block: 2, window: 2, dim: 6, hidden: 4, vocab: 4, inner: 40, innerEta: 0.06, lambda: 1e-4, x0: 0.05}
	rng := rand.New(rand.NewSource(5))
	m := newX2Model(cfg, rng)
	// Non-orthogonal embeddings, so the shared band makes A^T A off-diagonal
	// and the projector fills the private off-diagonal blocks.
	m.emb[0], m.emb[1] = 1, 0
	m.emb[2], m.emb[3] = 1, 0.5
	sc := newX2Scratch(cfg)
	block, w := cfg.block, cfg.window
	m.buildA([]int{0, 1}, sc.a)
	for col, sym := range []int{0, 1} {
		e := m.emb[sym*block : sym*block+block]
		for i := 0; i < block; i++ {
			if got := sc.a[(col*block+i)*w+col]; got != e[i] {
				t.Fatalf("private column %d row %d = %g, embedding %g", col, i, got, e[i])
			}
			if got := sc.a[(w*block+i)*w+col]; got != e[i] {
				t.Fatalf("shared column %d row %d = %g, embedding %g", col, i, got, e[i])
			}
		}
		other := 1 - col
		for i := 0; i < block; i++ {
			if got := sc.a[(other*block+i)*w+col]; got != 0 {
				t.Fatalf("column %d leaked into block %d row %d = %g", col, other, i, got)
			}
		}
	}
	solve := func(ctx []int) []float64 {
		m.buildA(ctx, sc.a)
		normalizeCols(sc.a, sc.ahat, sc.norms, cfg.dim, cfg.window)
		sc.solve(sc.ahat, cfg.innerEta, cfg.lambda, cfg.x0)
		return append([]float64(nil), sc.x...)
	}
	ab := solve([]int{0, 1})
	maxOff := 0.0
	for r := 0; r < block; r++ {
		for c := block; c < 2*block; c++ {
			if v := math.Abs(ab[r*cfg.dim+c]); v > maxOff {
				maxOff = v
			}
		}
	}
	if maxOff < 1e-3 {
		t.Fatalf("private off-diagonal block max %g", maxOff)
	}
	ba := solve([]int{1, 0})
	if diff := frobDiff(ab, ba); diff < 0.05 {
		t.Fatalf("swapped contexts gave ||X-X'|| = %g", diff)
	}
}

func TestX2SolvesConstraint(t *testing.T) {
	const d, w = 12, 6
	cfg := x2Config{dim: d, window: w, inner: 80, innerEta: 0.08, lambda: 1e-4, x0: 0.05}
	sc := newX2Scratch(cfg)
	rng := rand.New(rand.NewSource(2))
	raw := make([]float64, d*w)
	for i := range raw {
		raw[i] = rng.NormFloat64()
	}
	a := make([]float64, d*w)
	norms := make([]float64, w)
	normalizeCols(raw, a, norms, d, w)
	rel := sc.solve(a, cfg.innerEta, cfg.lambda, cfg.x0)
	if rel > 1e-2 {
		t.Fatalf("relative residual %g", rel)
	}
	p := x2Projector(t, a, d, w)
	if dist := frobDiff(sc.x, p) / frob(p); dist > 0.15 {
		t.Fatalf("distance to projector %g, residual %g", dist, rel)
	}
	eye := make([]float64, d*d)
	for i := 0; i < d; i++ {
		eye[i*d+i] = 1
	}
	if frobDiff(sc.x, eye) < 0.5 {
		t.Fatalf("solve collapsed to the identity, residual %g", rel)
	}
}

func TestX2StepGradient(t *testing.T) {
	cfg := x2Config{dim: 4, window: 2, hidden: 4, vocab: 3, inner: 1, innerEta: 0.05, lambda: 1e-4, x0: 0.05}
	sc := newX2Scratch(cfg)
	rng := rand.New(rand.NewSource(3))
	d, w := cfg.dim, cfg.window
	a := make([]float64, d*w)
	x := make([]float64, d*d)
	u := make([]float64, d*d)
	for i := range a {
		a[i] = rng.NormFloat64()
	}
	for i := range x {
		x[i] = 0.1 * rng.NormFloat64()
	}
	for i := range u {
		u[i] = rng.NormFloat64()
	}
	dx := make([]float64, d*d)
	da := make([]float64, d*w)
	sc.vjp(x, a, u, dx, da, cfg.innerEta, cfg.lambda)

	eps := 1e-6
	loss := func(x, a []float64) float64 {
		cp := append([]float64(nil), x...)
		sc.gdStep(cp, a, cfg.innerEta, cfg.lambda)
		s := 0.0
		for i := range u {
			s += u[i] * cp[i]
		}
		return s
	}
	baseX := append([]float64(nil), x...)
	baseA := append([]float64(nil), a...)
	for i := range x {
		x[i] = baseX[i] + eps
		lp := loss(x, a)
		x[i] = baseX[i] - eps
		lm := loss(x, a)
		x[i] = baseX[i]
		got := dx[i]
		want := (lp - lm) / (2 * eps)
		if math.Abs(got-want) > 1e-4 && math.Abs(got-want) > 1e-3*math.Max(math.Abs(want), 1) {
			t.Fatalf("dX[%d] = %g, numeric %g", i, got, want)
		}
	}
	copy(x, baseX)
	for i := range a {
		a[i] = baseA[i] + eps
		lp := loss(x, a)
		a[i] = baseA[i] - eps
		lm := loss(x, a)
		a[i] = baseA[i]
		got := da[i]
		want := (lp - lm) / (2 * eps)
		if math.Abs(got-want) > 1e-4 && math.Abs(got-want) > 1e-3*math.Max(math.Abs(want), 1) {
			t.Fatalf("dA[%d] = %g, numeric %g", i, got, want)
		}
	}
}

func TestX2EmbeddingGradient(t *testing.T) {
	cfg := x2Config{block: 2, dim: 6, window: 2, hidden: 8, vocab: 4, inner: 3, innerEta: 0.05, lambda: 1e-4, x0: 0.05}
	rng := rand.New(rand.NewSource(4))
	m := newX2Model(cfg, rng)
	sc := newX2Scratch(cfg)
	ctx := []int{0, 1}
	y := 2
	gEmb := make([]float64, len(m.emb))
	gW1 := make([]float64, len(m.w1))
	gB1 := make([]float64, len(m.b1))
	gW2 := make([]float64, len(m.w2))
	gB2 := make([]float64, len(m.b2))
	m.exampleGrad(ctx, y, gEmb, gW1, gB1, gW2, gB2, sc)

	lossAt := func() float64 {
		lg := m.logits(ctx, sc)
		return crossEntropy(lg, sc.dlogits, y)
	}
	eps := 1e-5
	check := func(name string, p, g []float64, indexes []int) {
		t.Helper()
		for _, i := range indexes {
			orig := p[i]
			p[i] = orig + eps
			lp := lossAt()
			p[i] = orig - eps
			lm := lossAt()
			p[i] = orig
			want := (lp - lm) / (2 * eps)
			got := g[i]
			if math.Abs(got-want) > 2e-3 && math.Abs(got-want) > 2e-2*math.Max(math.Abs(want), 1) {
				t.Fatalf("%s[%d] = %g, numeric %g", name, i, got, want)
			}
		}
	}
	check("emb", m.emb, gEmb, []int{0, 1, cfg.block, m.pad() * cfg.block})
	check("w1", m.w1, gW1, []int{0, 7, len(m.w1) - 1})
}

func TestColGradient(t *testing.T) {
	cfg := x2Config{block: 2, dim: 6, window: 2, hidden: 8, vocab: 4, inner: 1, innerEta: 0.05, lambda: 1e-4, x0: 0.05}
	rng := rand.New(rand.NewSource(4))
	m := newColModel(cfg, rng)
	sc := newX2Scratch(cfg)
	ctx := []int{0, 1}
	y := 2
	gEmb := make([]float64, len(m.emb))
	gW1 := make([]float64, len(m.w1))
	gB1 := make([]float64, len(m.b1))
	gW2 := make([]float64, len(m.w2))
	gB2 := make([]float64, len(m.b2))
	m.exampleGrad(ctx, y, gEmb, gW1, gB1, gW2, gB2, sc)

	lossAt := func() float64 {
		lg := m.logits(ctx, sc)
		return crossEntropy(lg, sc.dlogits, y)
	}
	eps := 1e-5
	for _, i := range []int{0, 1, cfg.block, m.cfg.vocab * cfg.block} {
		orig := m.emb[i]
		m.emb[i] = orig + eps
		lp := lossAt()
		m.emb[i] = orig - eps
		lm := lossAt()
		m.emb[i] = orig
		want := (lp - lm) / (2 * eps)
		got := gEmb[i]
		if math.Abs(got-want) > 2e-3 && math.Abs(got-want) > 2e-2*math.Max(math.Abs(want), 1) {
			t.Fatalf("emb[%d] = %g, numeric %g", i, got, want)
		}
	}
	i := 0
	orig := m.w1[i]
	m.w1[i] = orig + eps
	lp := lossAt()
	m.w1[i] = orig - eps
	lm := lossAt()
	m.w1[i] = orig
	want := (lp - lm) / (2 * eps)
	if math.Abs(gW1[i]-want) > 2e-3 && math.Abs(gW1[i]-want) > 2e-2*math.Max(math.Abs(want), 1) {
		t.Fatalf("w1[0] = %g, numeric %g", gW1[i], want)
	}
}

func TestXDropout(t *testing.T) {
	cfg := x2Config{block: 2, dim: 6, window: 2, hidden: 4, vocab: 4, inner: 3, innerEta: 0.05, lambda: 1e-4, x0: 0.05}
	rng := rand.New(rand.NewSource(4))
	m := newX2Model(cfg, rng)
	sc := newX2Scratch(cfg)
	sc.xMask = make([]float64, sc.nIn)
	for i := range sc.xMask {
		sc.xMask[i] = 1
	}
	sc.xMask[0] = 0
	sc.xMask[1] = 1.25
	ctx := []int{0, 1}
	y := 2
	gEmb := make([]float64, len(m.emb))
	gW1 := make([]float64, len(m.w1))
	gB1 := make([]float64, len(m.b1))
	gW2 := make([]float64, len(m.w2))
	gB2 := make([]float64, len(m.b2))
	m.exampleGrad(ctx, y, gEmb, gW1, gB1, gW2, gB2, sc)
	for h := 0; h < cfg.hidden; h++ {
		if g := gW1[h*sc.nIn]; g != 0 {
			t.Fatalf("dropped X entry w1[%d] = %g", h*sc.nIn, g)
		}
	}
	lossAt := func() float64 {
		lg := m.logits(ctx, sc)
		return crossEntropy(lg, sc.dlogits, y)
	}
	eps := 1e-5
	check := func(name string, p, g []float64, i int) {
		t.Helper()
		orig := p[i]
		p[i] = orig + eps
		lp := lossAt()
		p[i] = orig - eps
		lm := lossAt()
		p[i] = orig
		want := (lp - lm) / (2 * eps)
		if math.Abs(g[i]-want) > 2e-3 && math.Abs(g[i]-want) > 2e-2*math.Max(math.Abs(want), 1) {
			t.Fatalf("%s[%d] = %g, numeric %g", name, i, g[i], want)
		}
	}
	check("w1", m.w1, gW1, 1)
	check("emb", m.emb, gEmb, 0)
}

func TestX2Alphabet(t *testing.T) {
	text := []byte("ab ca\n")
	a := newX2Alphabet(text)
	if len(a.from) != 5 {
		t.Fatalf("alphabet %q", a.from)
	}
	enc := a.encode(text)
	if string(a.decode(enc)) != string(text) {
		t.Fatalf("roundtrip %q", a.decode(enc))
	}
	// A byte missing from the corpus becomes a space.
	got := a.decode(a.encode([]byte("z")))
	if string(got) != " " {
		t.Fatalf("unknown byte encoded to %q", got)
	}
}

func TestX2LearnsBigram(t *testing.T) {
	cfg := x2Config{block: 2, dim: 10, window: 4, hidden: 32, vocab: 4, inner: 24, innerEta: 0.1, lambda: 1e-4, x0: 0.05}
	rng := rand.New(rand.NewSource(1))
	m := newX2Model(cfg, rng)
	text := make([]byte, 600)
	for i := range text {
		if i < len(text)/2 {
			text[i] = byte(i % 2)
		} else {
			text[i] = byte(2 + i%2)
		}
	}
	loss, rel := m.train(text, 250, 24, 4e-3, rng, 0)
	if math.IsNaN(loss) {
		t.Fatal("loss is NaN")
	}
	sc := newX2Scratch(cfg)
	correct, total := 0, 0
	for pos := cfg.window; pos < len(text); pos++ {
		ctx := x2Context(text, pos, cfg.window, m.pad())
		if argmax(m.logits(ctx, sc)) == int(text[pos]) {
			correct++
		}
		total++
	}
	acc := float64(correct) / float64(total)
	if acc < 0.9 {
		t.Fatalf("accuracy %g (loss %g residual %g)", acc, loss, rel)
	}
}

func x2Projector(t *testing.T, a []float64, d, w int) []float64 {
	t.Helper()
	g := make([]float64, w*w)
	mulATInto(g, a, a, d, w, w)
	inv, ok := invertMatrix(g, w)
	if !ok {
		t.Fatal("singular gram matrix")
	}
	tmp := make([]float64, d*w)
	mulInto(tmp, a, inv, d, w, w)
	p := make([]float64, d*d)
	mulBTInto(p, tmp, a, d, w, d)
	return p
}

func invertMatrix(a []float64, n int) ([]float64, bool) {
	m := make([]float64, n*2*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			m[i*2*n+j] = a[i*n+j]
		}
		m[i*2*n+n+i] = 1
	}
	for col := 0; col < n; col++ {
		piv := col
		best := math.Abs(m[col*2*n+col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(m[r*2*n+col]); v > best {
				best, piv = v, r
			}
		}
		if best < 1e-12 {
			return nil, false
		}
		if piv != col {
			for j := 0; j < 2*n; j++ {
				m[col*2*n+j], m[piv*2*n+j] = m[piv*2*n+j], m[col*2*n+j]
			}
		}
		div := m[col*2*n+col]
		for j := 0; j < 2*n; j++ {
			m[col*2*n+j] /= div
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := m[r*2*n+col]
			for j := 0; j < 2*n; j++ {
				m[r*2*n+j] -= f * m[col*2*n+j]
			}
		}
	}
	inv := make([]float64, n*n)
	for i := 0; i < n; i++ {
		copy(inv[i*n:(i+1)*n], m[i*2*n+n:i*2*n+2*n])
	}
	return inv, true
}

func frob(a []float64) float64 {
	s := 0.0
	for _, v := range a {
		s += v * v
	}
	return math.Sqrt(s)
}

func frobDiff(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return math.Sqrt(s)
}
