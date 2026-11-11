"""Render Gatekeeper demo-video slides (1920x1080 PNGs) with PIL. v1."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "slides"
OUT.mkdir(exist_ok=True)
W, H = 1920, 1080

BG = (10, 16, 28)
FG = (230, 237, 243)
MUTED = (139, 148, 158)
ACCENT = (0, 180, 160)      # PayPal-ish teal
ACCENT2 = (88, 166, 255)    # blue
WARN = (240, 178, 90)
RED = (248, 81, 73)
GREEN = (63, 185, 80)
CARD = (18, 26, 40)
BORDER = (48, 54, 61)

SANS = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SANSB = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
MONOB = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"


def F(size, bold=False, mono=False):
    return ImageFont.truetype(MONOB if mono and bold else MONO if mono else SANSB if bold else SANS, size)


def wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def canvas():
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def footer(d, n):
    d.text((80, H - 60), "Gatekeeper — PayPal AI Hackathon", font=F(24), fill=MUTED)
    d.text((W - 130, H - 60), f"{n}/6", font=F(24), fill=MUTED)
    d.line([(80, H - 90), (W - 80, H - 90)], fill=BORDER, width=2)


def title(d, text, y=120, size=72, color=FG, x=100):
    d.text((x, y), text, font=F(size, bold=True), fill=color)


def bullets(d, items, y, size=38, gap=34, x=100, max_w=1650, color=FG, marker=">"):
    for it, c in items:
        d.text((x, y), marker, font=F(size, bold=True), fill=ACCENT)
        for ln in wrap(d, it, F(size), max_w - 60):
            d.text((x + 60, y), ln, font=F(size), fill=c or color)
            y += size + 14
        y += gap
    return y


def card(d, xy, fillc=CARD, border=BORDER, radius=18, width=2):
    d.rounded_rectangle(xy, radius=radius, fill=fillc, outline=border, width=width)


def stat(d, x, big, small, color=ACCENT, w=520):
    card(d, (x, 300, x + w, 540))
    d.text((x + 30, 330), big, font=F(60, bold=True), fill=color)
    for i, ln in enumerate(wrap(d, small, F(26), w - 60)):
        d.text((x + 30, 420 + i * 34), ln, font=F(26), fill=FG)


# S1: title
img, d = canvas()
d.text((100, 280), "Gatekeeper", font=F(150, bold=True), fill=FG)
d.text((104, 460), "The brakes for agentic commerce", font=F(52), fill=ACCENT)
d.text((104, 550), "An AI agent proposes a payment.\nRisk decides first: approve, hold, or block —\nbefore any money moves.", font=F(38), fill=MUTED)
card(d, (100, 760, 900, 890))
d.text((130, 788), "PayPal sandbox Orders v2 + Disputes v1", font=F(30, bold=True), fill=FG)
d.text((130, 832), "Real IEEE-CIS model, measured — not synthetic", font=F(26), fill=MUTED)
card(d, (1000, 760, 1760, 890))
d.text((1030, 788), "github.com/faketut/gatekeeper", font=F(30, bold=True), fill=FG)
d.text((1030, 832), "Public repo · Docker gateway · approval queue", font=F(26), fill=MUTED)
footer(d, 1)
img.save(OUT / "s1.png")

# S2: problem
img, d = canvas()
title(d, "Every agent demo wires a wallet. Nobody ships the brakes.")
bullets(d, [
    ("An AI shopping agent proposes a payment — and today, it just pays.", None),
    ("A normal purchase, a 3 a.m. order from a new device, a high-value burst: to a wallet, they all look the same.", None),
    ("Gatekeeper sits at the execute / capture choke point: unapproved or BLOCKed payments are refused.", ACCENT2),
], y=300, size=40, gap=40)
footer(d, 2)
img.save(OUT / "s2.png")

# S3: Demo 1 behavioural
img, d = canvas()
title(d, "Live demo: three proposals, three verdicts")
rows = [
    ("$49  ·  daytime", "5.7%", "ALLOW", GREEN, "Normal purchase, goes straight through."),
    ("$2,000  ·  3 a.m., new device", "45.0%", "REVIEW", WARN, "Held in the approval queue — nothing moves until a human says yes."),
    ("$12,000  ·  3 a.m. burst", "78.8%", "BLOCK", RED, "Blocked outright, before money moves."),
]
y = 260
for desc, score, verdict, col, note in rows:
    card(d, (100, y, 1820, y + 180))
    d.text((140, y + 25), desc, font=F(36, bold=True), fill=FG)
    d.text((140, y + 90), note, font=F(28), fill=MUTED)
    d.text((1400, y + 20), score, font=F(56, bold=True), fill=col)
    d.text((1400, y + 100), verdict, font=F(32, bold=True), fill=col)
    y += 210
d.text((100, y + 15), "Demo thresholds 0.35 / 0.75 — behavioural scorer only, NOT the trained model's thresholds.",
       font=F(28), fill=WARN)
footer(d, 3)
img.save(OUT / "s3.png")

# S4: real model v4
img, d = canvas()
title(d, "The real model: IEEE-CIS, full data, held-out test")
d.text((100, 230), "590,540 rows · 3.50% fraud · 70 / 15 / 15 split, seed 42 · LightGBM, tuned (v4)",
       font=F(30), fill=MUTED)
stat(d, 100, "0.9771", "ROC AUC\n(test n = 88,581)", ACCENT2)
stat(d, 660, "0.8711", "Average precision\n(3,099 fraud)", ACCENT2)
stat(d, 1220, "$6,587", "Gated loss per 10K\nvs $51,593 with no gate", ACCENT, w=600)
bullets(d, [
    ("REVIEW 0.0455: precision 44.4%, recall 90.5% — 7.1% of transactions reviewed.", None),
    ("BLOCK 0.5750: precision 92.9%, recall 74.8% — three quarters of fraud stopped outright.", None),
], y=600, size=36, gap=30)
footer(d, 4)
img.save(OUT / "s4.png")

# S5: PayPal reality
img, d = canvas()
title(d, "Not just a model demo: a real PayPal sandbox order")
bullets(d, [
    ("Orders v2: order created and verified CREATED, with an approve link.", None),
    ("Capture needs the payer's approval first — that is PayPal's lifecycle, not a failure.", None),
    ("Disputes v1: chargeback evidence is drafted for merchant review — never auto-submitted.", None),
    ("The browser demo scorer is behavioural; the real gateway runs via Docker from the repo.", ACCENT2),
], y=300, size=40, gap=40)
footer(d, 5)
img.save(OUT / "s5.png")

# S6: honest close
img, d = canvas()
title(d, "The honest weakness — and what we shipped anyway")
bullets(d, [
    ("Review precision is 44.4%: most reviewed transactions are legitimate.", None),
    ("That is the price of catching 90%+ of fraud at 3.5% prevalence — and it is half the review burden of the earlier model.", None),
    ("The v5 experiment did not beat v4, so it stayed out of the headline — and out of the shipped model.", None),
    ("Code, model, and numbers are public: github.com/faketut/gatekeeper", ACCENT),
], y=300, size=38, gap=36)
footer(d, 6)
img.save(OUT / "s6.png")

print("slides done:", sorted(p.name for p in OUT.glob("*.png")))
