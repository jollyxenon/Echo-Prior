#!/usr/bin/env python3
"""
Deterministically convert paper/paper1-icml-en.md into the official
ICML 2027 two-column LaTeX manuscript paper/paper1_preprint.tex.
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
EN_MD = HERE / "paper1-icml-en.md"
BIB_FILE = HERE / "references.bib"
OUT_TEX = HERE / "paper1_preprint.tex"

LATEX_TO_UTF8 = {
    '{\\"a}': "ä",
    '{\\"o}': "ö",
    '{\\"u}': "ü",
    '{\\"A}': "Ä",
    '{\\"O}': "Ö",
    '{\\"U}': "Ü",
    "{\\'e}": "é",
    "{\\'a}": "á",
    "{\\'\\i}": "í",
    "{\\'o}": "ó",
    "{\\'u}": "ú",
    "{\\'A}": "Á",
    "{\\'E}": "É",
    "{\\'c}": "ć",
    "{\\v{c}}": "č",
    "{\\v{z}}": "ž",
    "{\\v{s}}": "š",
    "{\\H{o}}": "ő",
    "{\\H{u}}": "ű",
    "{\\~n}": "ñ",
    "{\\ss}": "ß",
}


def build_bib_map(bib_text: str) -> dict[tuple[str, str], str]:
    """Build (author_lastname_lower, year) -> bib_key mapping from references.bib."""
    bib_map = {}
    for m in re.finditer(
        r"@\w+\{([^,]+),\s*author\s*=\s*\{(.+?)\},\s*title.*?year\s*=\s*\{(\d{4}[a-z]?)\}",
        bib_text,
        re.DOTALL,
    ):
        key, authors, year = m.groups()
        first_last = authors.split(",")[0].strip()
        for k_tex, v_utf in LATEX_TO_UTF8.items():
            first_last = first_last.replace(k_tex, v_utf)
        first_last = first_last.lower()
        bib_map[(first_last, year)] = key
        ascii_last = (
            first_last.replace("é", "e")
            .replace("á", "a")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ö", "o")
            .replace("ü", "u")
            .replace("ä", "a")
            .replace("ß", "ss")
            .replace("ő", "o")
            .replace("č", "c")
            .replace("ć", "c")
            .replace("ž", "z")
            .replace("ñ", "n")
        )
        bib_map[(ascii_last, year)] = key
    return bib_map


def convert() -> None:
    """Convert paper1-icml-en.md to ICML 2027 two-column paper1_preprint.tex."""
    bib_text = BIB_FILE.read_text(encoding="utf-8")
    bib_map = build_bib_map(bib_text)
    md_text = EN_MD.read_text(encoding="utf-8")

    # 1. Extract End Matter declarations from the end and place them right before @@BIBLIOGRAPHY@@
    part_before_ref, rem = md_text.split("\n## References\n", 1)
    _, part_app_and_end = rem.split("\n## Appendix A ", 1)
    part_app, part_end = part_app_and_end.split("\n---\n\n## Code and Data Availability\n", 1)

    full_md = (
        part_before_ref.rstrip()
        + "\n\n@@CLEARPAGE@@\n\n## Code and Data Availability\n"
        + part_end.rstrip()
        + "\n\n@@BIBLIOGRAPHY@@\n\n## Appendix A "
        + part_app.rstrip()
        + "\n"
    )

    # 2. Replace Mermaid Figure 1 block and caption
    fig1_pattern = re.compile(
        r"\*\*Figure 1:\s*(.+?)\*\*\s*(.*?)\n+```mermaid\n.*?\n```",
        re.DOTALL,
    )

    def repl_fig1(m: re.Match) -> str:
        bold_title, rest_cap = m.groups()
        rest_str = f" {rest_cap.strip()}" if rest_cap.strip() else ""
        return f"@@FIGURE1_START@@\n\\textbf{{{bold_title.strip()}}}{rest_str}\n@@FIGURE1_END@@"

    full_md = fig1_pattern.sub(repl_fig1, full_md)

    # 3. Replace HTML <img> + **Figure X: ...** for Figures 2, 3, etc.
    fig_img_pattern = re.compile(
        r'<img\s+src="figures/([^"]+)\.png"[^>]*>\s*\n+\*\*Figure\s+(\d+):\s*(.+?)\*\*\s*(.+?)(?=\n\n)',
        re.DOTALL,
    )

    def repl_fig(m: re.Match) -> str:
        stem, fnum, bold_title, rest_cap = m.groups()
        return f"@@FIGURE_START:{fnum}:{stem}@@\n\\textbf{{{bold_title.strip()}}} {rest_cap.strip()}\n@@FIGURE_END@@"

    full_md = fig_img_pattern.sub(repl_fig, full_md)

    # 4. Protect math blocks $$ ... $$ and inline math $ ... $
    math_store: list[str] = []

    def save_math(tex_str: str) -> str:
        idx = len(math_store)
        math_store.append(tex_str)
        return f"@@MATH{idx}@@"

    def repl_display_math(m: re.Match) -> str:
        inner = m.group(1).strip()
        return save_math(f"\\[\n{inner}\n\\]")

    full_md = re.sub(r"\$\$(.+?)\$\$", repl_display_math, full_md, flags=re.DOTALL)

    def repl_inline_math(m: re.Match) -> str:
        inner = m.group(1)
        inner = inner.replace("\\rightarrow", "\\rightarrow\\allowbreak")
        return save_math(f"${inner}$")

    full_md = re.sub(r"(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)", repl_inline_math, full_md)

    # 5. Convert Citations (Parenthetical & Narrative)
    full_md = full_md.replace(
        "(unlike Warm-Start QAOA, see Egger et al., 2021; Tate et al., 2023 and Appendix D.2)",
        "(unlike Warm-Start QAOA, see \\citealp{egger2021warm,tate2023warm} and Appendix D.2)",
    )

    cite_item_re = re.compile(
        r"^([A-Z][a-zA-Zéöüäß\-\x27]+)(?:\s+(?:et al\.|(?:&|and)\s+[A-Z][a-zA-Zéöüäß\-\x27]+))?,\s*((?:19|20)\d{2}[a-z]?)$"
    )

    def repl_paren_cite(m: re.Match) -> str:
        content = m.group(1)
        parts = [p.strip() for p in content.split(";")]
        keys = []
        for p in parts:
            cm = cite_item_re.match(p)
            if not cm:
                return m.group(0)
            last, yr = cm.group(1).lower(), cm.group(2)
            key = bib_map.get((last, yr))
            if not key:
                raise ValueError(f"Unmapped citation: {p}")
            keys.append(key)
        return f"\\citep{{{','.join(keys)}}}"

    full_md = re.sub(r"\(([^()]*?\b(?:19|20)\d{2}[a-z]?[^()]*?)\)", repl_paren_cite, full_md)

    narr_re = re.compile(
        r"\b([A-Z][a-zA-Zéöüäß\-\x27]+)(?:\s+et al\.|\s+(?:and|&)\s+[A-Z][a-zA-Zéöüäß\-\x27]+)?\s*\(((?:19|20)\d{2}[a-z]?)\)"
    )

    def repl_narr_cite(m: re.Match) -> str:
        raw_last, yr = m.group(1), m.group(2)
        if raw_last.endswith("'s"):
            poss, last = "'s", raw_last[:-2].lower()
        elif raw_last.endswith("'"):
            poss, last = "'", raw_last[:-1].lower()
        else:
            poss, last = "", raw_last.lower()
        key = bib_map.get((last, yr))
        if not key:
            raise ValueError(f"Unmapped narrative citation: {m.group(0)}")
        return f"\\citet{{{key}}}{poss}"

    full_md = narr_re.sub(repl_narr_cite, full_md)

    # 6. Convert Markdown links [text](url) -> \url{url}
    full_md = re.sub(r"\[https?://[^\]]+\]\((https?://[^)]+)\)", r"\\url{\1}", full_md)

    # 7. Convert inline code `...` -> \texttt{...}
    code_store: list[str] = []

    def repl_inline_code(m: re.Match) -> str:
        c = m.group(1)
        c = (
            c.replace("\\", "\\textbackslash{}")
            .replace("_", "\\_\\allowbreak{}")
            .replace("/", "/\\allowbreak{}")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("#", "\\#")
            .replace("*", "{*}")
        )
        idx = len(code_store)
        code_store.append(f"\\texttt{{{c}}}")
        return f"@@CODE{idx}@@"

    full_md = re.sub(r"`([^`\n]+)`", repl_inline_code, full_md)

    # 8. Extract and convert Tables (**Table X: ...** + | ... |)
    table_pattern = re.compile(
        r"\*\*Table\s+([0-9A-Za-z\.]+):\s*(.+?)\*\*\s*\n+((?:\|[^\n]+\n)+)",
        re.DOTALL,
    )

    table_store: list[str] = []

    def format_inline_md(s: str) -> str:
        s = s.replace("<br>", " ").replace("\\*", "@@AST@@")
        s = s.replace("%", "\\%").replace("&", "\\&")
        s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
        s = re.sub(r"\*(.+?)\*", r"\\emph{\1}", s)
        s = re.sub(r"_([^_]+)_", r"\\emph{\1}", s)
        s = s.replace("@@AST@@", "*")
        s = re.sub(r'"([^"]+)"', r"``\1''", s)
        s = s.replace("—", "---").replace("–", "--")
        s = (
            s.replace("é", "\\'e")
            .replace("ő", "\\H{o}")
            .replace("ü", '\\"u')
            .replace("ä", '\\"a')
            .replace("č", "\\v{c}")
        )
        return s

    def format_heading_md(s: str) -> str:
        res = format_inline_md(s)

        def repl_hmath(m: re.Match) -> str:
            midx = int(m.group(1))
            mtex = math_store[midx]
            plain = mtex.strip("$").replace("\\text{--}", "--").replace("\\", "")
            return f"\\texorpdfstring{{{mtex}}}{{{plain}}}"

        res = re.sub(r"@@MATH(\d+)@@", repl_hmath, res)
        return res

    def repl_table(m: re.Match) -> str:
        t_id, t_cap, t_body = m.groups()
        t_cap_tex = format_inline_md(t_cap.strip())
        lines = [l.strip() for l in t_body.strip().split("\n") if l.strip()]
        header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
        align_cells = [c.strip() for c in lines[1].strip("|").split("|")]
        data_rows = [
            [c.strip() for c in r.strip("|").split("|")] for r in lines[2:]
        ]

        col_specs = []
        for a in align_cells:
            if a.startswith(":") and a.endswith(":"):
                col_specs.append("c")
            elif a.endswith(":"):
                col_specs.append("r")
            else:
                col_specs.append("l")

        label_id = t_id.lower().replace(".", "_")
        is_appendix_tbl = t_id.startswith(("A", "B", "C", "D"))
        env_name = "table" if is_appendix_tbl else "table*"
        pos_spec = "[htbp]" if is_appendix_tbl else "[t]"

        if t_id == "1":
            col_spec_str = ">{\\raggedright\\arraybackslash}p{3.0cm}>{\\raggedright\\arraybackslash}p{5.0cm}>{\\raggedright\\arraybackslash}p{4.3cm}>{\\raggedright\\arraybackslash}p{7.8cm}"
        elif t_id in ("2", "D.1", "D.2"):
            col_spec_str = "lcccccc"
        elif t_id in ("3", "D.4"):
            col_spec_str = "llccccc"
        elif t_id == "4":
            col_spec_str = "llcccc"
        elif t_id == "5":
            col_spec_str = "cccccc"
        elif t_id == "D.3":
            col_spec_str = "cccccc"
        else:
            col_spec_str = "".join(col_specs)

        use_resize = t_id in ("1", "2", "3", "4", "5", "D.4")
        if t_id == "2":
            resize_w = "0.78\\textwidth"
        elif t_id in ("3", "D.4"):
            resize_w = "0.94\\textwidth"
        elif t_id == "4":
            resize_w = "0.86\\textwidth"
        elif t_id == "5":
            resize_w = "0.88\\textwidth"
        else:
            resize_w = "\\textwidth"
        tabcol_pt = "5.5pt" if t_id in ("3", "4", "5", "D.3", "D.4") else "4.0pt"
        tex_lines = [
            f"\\begin{{{env_name}}}{pos_spec}",
            "\\vskip -0.04in",
            f"\\caption{{{t_cap_tex}}}",
            f"\\label{{tab:{label_id}}}",
            "\\vskip 0.01in",
            "\\centering",
            "\\footnotesize",
            "\\renewcommand{\\arraystretch}{0.96}",
            f"\\setlength{{\\tabcolsep}}{{{tabcol_pt}}}",
        ]
        if use_resize:
            tex_lines.append(f"\\resizebox{{{resize_w}}}{{!}}{{%")
        tex_lines.append(f"\\begin{{tabular}}{{{col_spec_str}}}")
        tex_lines.append("\\toprule")
        h_tex = " & ".join(format_inline_md(c) for c in header_cells) + " \\\\"
        tex_lines.append(h_tex)
        tex_lines.append("\\midrule")
        for r_idx, row in enumerate(data_rows):
            if t_id in ("3", "D.4") and r_idx > 0 and row[0]:
                tex_lines.append("\\midrule")
            r_tex = " & ".join(format_inline_md(c) for c in row) + " \\\\"
            tex_lines.append(r_tex)
        tex_lines.append("\\bottomrule")
        tex_lines.append("\\end{tabular}%")
        if use_resize:
            tex_lines.append("}")
        tex_lines.append("\\vskip -0.02in")
        tex_lines.append(f"\\end{{{env_name}}}")

        idx = len(table_store)
        table_store.append("\n".join(tex_lines))
        return f"@@TABLE{idx}@@"

    full_md = table_pattern.sub(repl_table, full_md)

    # 9. Handle Propositions and Proofs
    prop_all_re = re.compile(
        r"\*\*Proposition\s+([1-5])\s*\(([^)]+)\)\.\*\*\s*(.+?)(?=\n\n\*\*(?:Proof|Construction))",
        re.DOTALL,
    )

    def repl_prop_all(m: re.Match) -> str:
        pnum, ptitle, pbody = m.groups()
        pbody_clean = re.sub(r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)", r"\1", pbody)
        pbody_clean = re.sub(r"(?<!_)_(?!_)([^_\n]+?)(?<!_)_(?!_)", r"\1", pbody_clean)
        return (
            f"\\begin{{proposition}}[{ptitle.strip()}]\n"
            f"\\label{{prop:{pnum}}}\n"
            f"{pbody_clean.strip()}\n"
            f"\\end{{proposition}}"
        )

    full_md = prop_all_re.sub(repl_prop_all, full_md)

    square_tokens = [
        f"@@MATH{i}@@" for i, m in enumerate(math_store) if m.strip() == "$\\square$"
    ]
    for st in square_tokens:
        full_md = full_md.replace(st, "@@QED@@")

    proof_re = re.compile(
        r"\*\*(Proof|Proof Sketch|Construction and Proof Sketch)\.\*\*\s*(.+?)\s*@@QED@@",
        re.DOTALL,
    )

    def repl_proof(m: re.Match) -> str:
        ptype, pbody = m.groups()
        if ptype == "Proof":
            return f"\\begin{{proof}}\n{pbody.strip()}\n\\end{{proof}}"
        return f"\\begin{{proof}}[{ptype}]\n{pbody.strip()}\n\\end{{proof}}"

    full_md = proof_re.sub(repl_proof, full_md)

    # 10. Process Line-by-Line Structure (Sections, Lists, Paragraphs, Figures)
    lines = full_md.split("\n")
    out_lines: list[str] = []
    in_abstract = False
    list_stack: list[tuple[str, int]] = []

    def close_lists_to(target_indent: int) -> None:
        while list_stack and list_stack[-1][1] >= target_indent:
            env, _ = list_stack.pop()
            out_lines.append(f"\\end{{{env}}}")

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        s = raw_line.strip()

        if s.startswith("# Echo-QAOA:"):
            i += 1
            continue

        if s == "---":
            close_lists_to(0)
            i += 1
            continue

        if s == "## Abstract":
            close_lists_to(0)
            out_lines.append("\\begin{abstract}")
            in_abstract = True
            i += 1
            continue

        if s.startswith("## "):
            close_lists_to(0)
            if in_abstract:
                out_lines.append("\\end{abstract}\n")
                in_abstract = False

            sec_title = s[3:].strip()
            if sec_title.startswith("Appendix A"):
                out_lines.append(
                    "\\newpage\n\\appendix\n\\onecolumn\n"
                    "\\counterwithin{table}{section}\n"
                    "\\counterwithin{figure}{section}"
                )
                clean_t = re.sub(r"^Appendix\s+[A-Z]\s+", "", sec_title)
                out_lines.append(f"\\section{{{format_heading_md(clean_t)}}}\\label{{app:A}}")
            elif sec_title.startswith("Appendix B"):
                clean_t = re.sub(r"^Appendix\s+[A-Z]\s+", "", sec_title)
                out_lines.append(f"\\section{{{format_heading_md(clean_t)}}}\\label{{app:B}}")
            elif sec_title.startswith("Appendix C"):
                clean_t = re.sub(r"^Appendix\s+[A-Z]\s+", "", sec_title)
                out_lines.append(f"\\section{{{format_heading_md(clean_t)}}}\\label{{app:C}}")
            elif sec_title.startswith("Appendix D"):
                clean_t = re.sub(r"^Appendix\s+[A-Z]\s+", "", sec_title)
                out_lines.append(f"\\section{{{format_heading_md(clean_t)}}}\\label{{app:D}}")
            else:
                m_sec = re.match(r"^(\d+)\s+(.+)$", sec_title)
                if m_sec:
                    snum, stitle = m_sec.groups()
                    out_lines.append(f"\\section{{{format_heading_md(stitle)}}}\\label{{sec:{snum}}}")
                else:
                    out_lines.append(f"\\section*{{{format_heading_md(sec_title)}}}")
            i += 1
            continue

        if s.startswith("### "):
            close_lists_to(0)
            sub_title = s[4:].strip()
            m_sub = re.match(r"^([0-9A-Z]+\.\d+)\s+(.+)$", sub_title)
            if m_sub:
                subnum, subtitle = m_sub.groups()
                lbl = subnum.lower().replace(".", "_")
                htex = format_heading_md(subtitle)
                out_lines.append(f"\\subsection{{{htex}}}\\label{{subsec:{lbl}}}")
            else:
                out_lines.append(f"\\subsection{{{format_heading_md(sub_title)}}}")
            i += 1
            continue

        if s.startswith("#### "):
            close_lists_to(0)
            subsub = s[5:].strip()
            m_ss = re.match(r"^(\d+)\.\s+(.+)$", subsub)
            if m_ss:
                _, sstitle = m_ss.groups()
                out_lines.append(f"\\subsubsection{{{format_heading_md(sstitle)}}}")
            else:
                out_lines.append(f"\\subsubsection{{{format_heading_md(subsub)}}}")
            i += 1
            continue

        if s == "@@CLEARPAGE@@":
            close_lists_to(0)
            out_lines.append("\\clearpage")
            i += 1
            continue

        if s == "@@BIBLIOGRAPHY@@":
            close_lists_to(0)
            out_lines.append("\\bibliography{references}")
            out_lines.append("\\bibliographystyle{icml2027}")
            i += 1
            continue

        if s == "@@FIGURE1_START@@":
            close_lists_to(0)
            cap_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "@@FIGURE1_END@@":
                cap_lines.append(lines[i].strip())
                i += 1
            cap_tex = format_inline_md(" ".join(cap_lines))
            out_lines.append(
                "\\begin{figure*}[t]\n"
                "\\vskip -0.02in\n"
                "\\begin{center}\n"
                "\\centerline{\\includegraphics[width=0.86\\textwidth]{figures/fig1_echo_qaoa_overview.pdf}}\n"
                "\\vskip -0.06in\n"
                f"\\caption{{{cap_tex}}}\n"
                "\\label{fig:1}\n"
                "\\end{center}\n"
                "\\vskip -0.18in\n"
                "\\end{figure*}"
            )
            i += 1
            continue

        if s.startswith("@@FIGURE_START:"):
            close_lists_to(0)
            _, fnum, stem = s.strip("@").split(":")
            cap_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "@@FIGURE_END@@":
                cap_lines.append(lines[i].strip())
                i += 1
            cap_tex = format_inline_md(" ".join(cap_lines))
            fig_w = "0.58\\textwidth" if fnum == "3" else "0.74\\textwidth"
            out_lines.append(
                f"@@FIGBLOCK{fnum}START@@\n"
                "\\begin{figure*}[t]\n"
                "\\vskip -0.02in\n"
                "\\begin{center}\n"
                f"\\centerline{{\\includegraphics[width={fig_w}]{{figures/{stem}.pdf}}}}\n"
                "\\vskip -0.06in\n"
                f"\\caption{{{cap_tex}}}\n"
                f"\\label{{fig:{fnum}}}\n"
                "\\end{center}\n"
                "\\vskip -0.24in\n"
                "\\end{figure*}\n"
                f"@@FIGBLOCK{fnum}END@@"
            )
            i += 1
            continue

        if re.match(r"^@@TABLE\d+@@$", s):
            close_lists_to(0)
            out_lines.append(s)
            i += 1
            continue

        m_enum = re.match(r"^(\s*)\d+\.\s+(.+)$", raw_line)
        m_item = re.match(r"^(\s*)-\s+(.+)$", raw_line)
        if m_enum or m_item:
            indent = len((m_enum or m_item).group(1))
            content = (m_enum or m_item).group(2)
            env_type = "enumerate" if m_enum else "itemize"

            while list_stack and list_stack[-1][1] > indent:
                env, _ = list_stack.pop()
                out_lines.append(f"\\end{{{env}}}")
            if not list_stack or list_stack[-1][1] < indent:
                list_stack.append((env_type, indent))
                out_lines.append(
                    f"\\begin{{{env_type}}}\\setlength{{\\itemsep}}{{0pt}}\\setlength{{\\parskip}}{{1pt}}\\setlength{{\\topsep}}{{1pt}}"
                )
            elif list_stack[-1][0] != env_type:
                env, _ = list_stack.pop()
                out_lines.append(f"\\end{{{env}}}")
                list_stack.append((env_type, indent))
                out_lines.append(
                    f"\\begin{{{env_type}}}\\setlength{{\\itemsep}}{{0pt}}\\setlength{{\\parskip}}{{1pt}}\\setlength{{\\topsep}}{{1pt}}"
                )

            out_lines.append(f"\\item {format_inline_md(content)}")
            i += 1
            continue

        if not s:
            out_lines.append("")
            i += 1
            continue

        if list_stack and raw_line.startswith("   "):
            out_lines.append(format_inline_md(s))
            i += 1
            continue

        if list_stack:
            close_lists_to(0)

        out_lines.append(format_inline_md(s))
        i += 1

    close_lists_to(0)

    # Combine Table 4 (table_store[3]) and Table 5 (table_store[4]) into a single table* float
    if len(table_store) >= 5:
        t4_inner = table_store[3]
        t5_inner = table_store[4]
        t4_inner = re.sub(r"\\vskip -0\.02in\n\\end\{table\*\}\s*$", "", t4_inner)
        t5_inner = re.sub(
            r"^\\begin\{table\*\}\[t\]\n\\vskip -0\.04in\n",
            lambda _: "\\vskip 0.02in\n",
            t5_inner,
        )
        t5_inner = re.sub(
            r"\\vskip -0\.02in(\n\\end\{table\*\}\s*)$",
            r"\\vskip -0.10in\1",
            t5_inner,
        )
        table_store[3] = t4_inner + "\n" + t5_inner
        table_store[4] = ""

    body_tex = "\n".join(out_lines)

    # Pre-queue two-column floats 1 section earlier to compensate for LaTeX's 1-page deferral of figure*/table*
    def move_token_before(tex: str, token: str, anchor: str) -> str:
        if token not in tex or anchor not in tex:
            return tex
        tex_removed = tex.replace(token, "", 1)
        return tex_removed.replace(anchor, token + "\n\n" + anchor, 1)

    # Table 0 = Table 1, Table 1 = Table 2, Table 2 = Table 3, Table 3 = Table 4+5
    def move_figblock_before(tex: str, fnum: str, anchor: str) -> str:
        pat = rf"@@FIGBLOCK{fnum}START@@\n(.*?)\n@@FIGBLOCK{fnum}END@@"
        m = re.search(pat, tex, flags=re.DOTALL)
        if not m:
            return tex
        inner = m.group(1)
        tex_removed = tex[: m.start()] + tex[m.end() :]
        return tex_removed.replace(anchor, inner + "\n\n" + anchor, 1)

    body_tex = move_token_before(body_tex, "@@TABLE0@@", r"\subsection{Single-Qubit Amplitude")
    body_tex = move_figblock_before(body_tex, "2", r"\subsection{Strict Convexity")
    body_tex = move_token_before(body_tex, "@@TABLE1@@", r"\subsection{Strict Convexity")
    body_tex = move_figblock_before(body_tex, "3", r"\subsection{Local Sensitivity")
    body_tex = move_token_before(body_tex, "@@TABLE2@@", r"\subsection{Offline Probability Teachers")
    body_tex = move_token_before(body_tex, "@@TABLE3@@", r"\subsection{Experimental Setup")
    body_tex = body_tex.replace("@@TABLE4@@", "")

    # 11. Restore Tables, Code, and Math tokens
    body_tex = body_tex.replace("@@QED@@", "\\hfill$\\square$")
    for idx, tbl_tex in enumerate(table_store):
        body_tex = body_tex.replace(f"@@TABLE{idx}@@", tbl_tex)

    for idx, c_tex in enumerate(code_store):
        body_tex = body_tex.replace(f"@@CODE{idx}@@", c_tex)

    for idx, m_tex in enumerate(math_store):
        if m_tex.startswith("\\[\n"):
            body_tex = re.sub(
                rf"\n*@@MATH{idx}@@\n*",
                lambda _, s=m_tex: f"\n{s}\n",
                body_tex,
            )
        else:
            body_tex = body_tex.replace(f"@@MATH{idx}@@", m_tex)

    # 12. Assemble full ICML 2027 two-column LaTeX document
    preamble = r"""%%%%%%%% ICML 2027 LATEX MANUSCRIPT: ECHO-QAOA %%%%%%%%%%%%%%%%%
%
% Deterministic two-column ICML 2027 conversion of paper/paper1-icml-en.md.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\documentclass{article}

\usepackage[T1]{fontenc}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{array}
\usepackage{hyperref}
\def\UrlBreaks{\do\/\do-\do_}

\newcommand{\theHalgorithm}{\arabic{algorithm}}

\usepackage[preprint]{icml2027}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}
\usepackage{amsthm}
\usepackage[capitalize,noabbrev]{cleveref}

\emergencystretch=3em
\binoppenalty=700
\relpenalty=500
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\dblfloatpagefraction}{0.85}
\setcounter{topnumber}{4}
\setcounter{bottomnumber}{4}
\setcounter{totalnumber}{6}
\setcounter{dbltopnumber}{4}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THEOREMS & PROPOSITIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\makeatletter
\def\thm@space@setup{%
  \thm@preskip=2pt plus 1pt minus 1pt
  \thm@postskip=2pt plus 1pt minus 1pt
}
\renewenvironment{proof}[1][\proofname]{\par
  \pushQED{\qed}%
  \normalfont \topsep1pt plus1pt\relax
  \trivlist
  \item[\hskip\labelsep
        \itshape
    #1\@addpunct{.}]\ignorespaces
}{%
  \popQED\endtrivlist\@endpefalse
}
\makeatother
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}{Corollary}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{assumption}{Assumption}
\newtheorem{remark}{Remark}

\setlength{\parskip}{1.4pt plus 0.5pt minus 0.4pt}
\setlength{\leftmargini}{1.25em}
\abovedisplayskip 2.5pt plus 1pt minus 1pt
\belowdisplayskip 2.5pt plus 1pt minus 1pt
\abovedisplayshortskip 0pt plus 1pt
\belowdisplayshortskip 1.5pt plus 1pt minus 1pt
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\dblfloatsep}{6pt plus 2pt minus 2pt}

\icmltitlerunning{Echo-QAOA: Constructive Spin-Echo Response Design and Per-Qubit Marginal Distillation}

\begin{document}

\twocolumn[
\icmltitle{Echo-QAOA: Constructive Spin-Echo Response Design and \\Per-Qubit Marginal Distillation for Variational Quantum Optimization}

\begin{icmlauthorlist}
\icmlauthor{Jianxuan Yao}{sht}
\end{icmlauthorlist}

\icmlaffiliation{sht}{ShanghaiTech University, Shanghai, China}
\icmlcorrespondingauthor{Jianxuan Yao}{yaojx2025@shanghaitech.edu.cn}

\icmlkeywords{Quantum Approximate Optimization Algorithm, Spin Echo, Knowledge Distillation, Combinatorial Optimization}

\vskip 0.3in
]

\printAffiliationsAndNotice{}
\hypersetup{pdftitle={Echo-QAOA: Constructive Spin-Echo Response Design and Per-Qubit Marginal Distillation for Variational Quantum Optimization},pdfauthor={Jianxuan Yao}}
"""

    full_tex = preamble + "\n" + body_tex + "\n\n\\end{document}\n"
    non_ascii = [c for c in full_tex if ord(c) >= 128]
    assert not non_ascii, f"Found non-ASCII chars in paper1_preprint.tex: {non_ascii}"
    OUT_TEX.write_text(full_tex, encoding="utf-8")
    print(f"Successfully wrote {len(full_tex.splitlines())} pure-ASCII lines to {OUT_TEX}")


if __name__ == "__main__":
    convert()
