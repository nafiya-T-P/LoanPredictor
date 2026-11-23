"""HTML building blocks for the cards, bars and confusion matrix.
(Replaces the old matplotlib helpers. Styling lives in styles.css.)"""
from html import escape


def bar_rows(rows, color="teal"):
    """rows = [(label, percent_0_to_100, text_shown_on_the_right), ...]"""
    out = []
    for label, pct, shown in rows:
        pct = max(0, min(100, pct))
        out.append(
            f'<div class="brow"><span class="bl">{escape(label)}</span>'
            f'<span class="bt"><span class="bf {color}" style="width:{pct:.0f}%"></span></span>'
            f'<span class="bv">{escape(str(shown))}</span></div>')
    return "".join(out)


def card(title, body, note=""):
    n = f'<div class="cnote">{escape(note)}</div>' if note else ""
    return f'<div class="card"><h3>{escape(title)}</h3>{body}{n}</div>'


def stat(value, label):
    return f'<div class="card stat"><div class="sv">{escape(value)}</div><div class="sl">{escape(label)}</div></div>'


def grid(cls, items):
    return f'<div class="grid {cls}">{"".join(items)}</div>'


def confusion(cm):
    """cm = [[TN, FP], [FN, TP]] (rows = actual Rejected/Approved)."""
    (tn, fp), (fn, tp) = cm
    r0, r1 = max(tn + fp, 1), max(fn + tp, 1)

    def cell(n, pct, dark):
        return (f'<div class="cmc {"dark" if dark else "light"}"><b>{n:,}</b>'
                f'<span>{pct:.0%}</span></div>')

    return ('<div class="cm">'
            '<div></div><div class="cmh">Pred. Rejected</div><div class="cmh">Pred. Approved</div>'
            '<div class="cml">Actual<br>Rejected</div>' + cell(tn, tn / r0, True) + cell(fp, fp / r0, False) +
            '<div class="cml">Actual<br>Approved</div>' + cell(fn, fn / r1, False) + cell(tp, tp / r1, True) +
            '</div>')