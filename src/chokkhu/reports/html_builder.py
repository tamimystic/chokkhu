from __future__ import annotations

import base64
import os

from chokkhu.core.logger import Logger


class HTMLReportBuilder:
    @staticmethod
    def build(
        save_dir: str, title: str = "Chokkhu EDA Report", max_inline_images: int = 50
    ):
        Logger.info(f"Generating HTML Report in {save_dir}...")
        image_files = [f for f in os.listdir(save_dir) if f.endswith(".png")]
        image_files.sort()
        header = (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{title}</title>\n"
            "<style>\n"
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
            "background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }\n"
            "h1 { text-align: center; color: #38bdf8; margin-bottom: 30px; font-weight: 700; letter-spacing: -0.5px; }\n"
            ".container { max-width: 1200px; margin: 0 auto; }\n"
            ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 24px; }\n"
            ".plot-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; text-align: center; }\n"
            ".plot-card:hover { transform: translateY(-2px); }\n"
            ".plot-card img { max-width: 100%; height: auto; border-radius: 8px; background: #fff; }\n"
            ".plot-title { font-size: 1.1em; font-weight: 600; margin-bottom: 15px; color: #f8fafc; }\n"
            ".badge { display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 0.8em; background: #0369a1; color: #bae6fd; }\n"
            "</style>\n</head>\n<body>\n"
            f'<div class="container">\n<h1>{title}</h1>\n'
            f'<p style="text-align:center; color:#94a3b8; margin-bottom:30px;">'
            f"Generated with Chokkhu AI &bull; {len(image_files)} Visualizations</p>\n"
            f'<div class="grid">\n'
        )
        html_content = header
        for img_file in image_files[:max_inline_images]:
            img_path = os.path.join(save_dir, img_file)
            with open(img_path, "rb") as img_f:
                encoded_string = base64.b64encode(img_f.read()).decode("utf-8")
            display_name = img_file.replace(".png", "").replace("_", " ").title()
            html_content += (
                f'<div class="plot-card">\n'
                f'<span class="badge">Visualization</span>\n'
                f'<div class="plot-title">{display_name}</div>\n'
                f'<img src="data:image/png;base64,{encoded_string}" alt="{display_name}" loading="lazy">\n'
                f"</div>\n"
            )
        html_content += "</div>\n</div>\n</body>\n</html>\n"
        report_path = os.path.join(save_dir, "chokkhu_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        Logger.info(f"HTML Report generated successfully: {report_path}")
