import gradio as gr
import pdfkit
import tempfile
import shutil
import torch
import pandas as pd
import os
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

# ========== Backend Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Progress indicator for model loading
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
set_seed(42)
print("Model loaded successfully!")

# ========== Data Initialization ==========
# Dummy label maps (update yours if needed)
label_map = {}
if os.path.exists("label_map.csv"):
    df_map = pd.read_csv("label_map.csv", header=None)
    label_map = {str(k): v for k, v in zip(df_map[0], df_map[1])}
else:
    print("Warning: label_map.csv not found, using empty mapping")

# Cache directory for PDFs
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

attack_source_map = {
    "Bot": "ml", "Infiltration": "ml", "XSS": "ml", "Web Attack": "ml", "Brute Force": "ml",
    "BENIGN": "fl", "DDoS": "fl", "DoS": "fl", "DoS GoldenEye": "fl", "DoS Hulk": "fl",
    "DoS slowloris": "fl", "FTP-Patator": "fl", "SSH-Patator": "fl", "PortScan": "fl", "Port Scan": "fl"
}

attack_explanations = {
    "BOT": "Automated traffic often used to scan networks or launch spam/DDoS attacks.",
    "INFILTRATION": "Unauthorized system access to steal or manipulate sensitive data.",
    "XSS": "Injects malicious scripts into trusted websites, often for stealing credentials.",
    "WEB ATTACK": "Targets web apps using techniques like SQLi, XSS, etc.",
    "BRUTE FORCE": "Tries all password combinations to break into accounts.",
    "BENIGN": "Normal, safe traffic with no threat patterns.",
    "DDOS": "Overwhelms a server by flooding it with traffic from multiple sources.",
    "DOS": "Disrupts services by overwhelming a system with traffic or requests.",
    "DOS GOLDENEYE": "Sends slow HTTP requests to keep server threads occupied.",
    "DOS HULK": "Generates excessive HTTP requests to crash web servers.",
    "DOS SLOWLORIS": "Sends partial HTTP headers slowly to exhaust server sockets.",
    "FTP-PATATOR": "Brute force attacks on FTP servers to guess credentials.",
    "SSH-PATATOR": "Same as above, but for SSH login.",
    "PORTSCAN": "Scans open ports to detect vulnerabilities.",
    "PORT SCAN": "Scans network ports to discover attack surfaces."
}

# All available attack options for dropdown
ALL_ATTACK_TYPES = sorted(list(attack_explanations.keys()))

# ========== Backend Functions ==========
def decode_label(label):
    """Normalize and decode label from numeric or text form"""
    raw = str(label).strip()
    if raw in label_map:
        return label_map[raw]
    for k, v in label_map.items():
        if raw.lower() == v.lower() or raw.lower() == k.lower():
            return v
    return raw

def sanitize_filename(name):
    """Create safe filenames for downloads"""
    return "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in name)

def generate_llm_insight(predicted_label, detailed=True):
    """Generate AI insight about an attack type"""
    label = predicted_label.strip().upper()
    
    # Generate appropriate prompt based on attack type
    if label == "BENIGN":
        prompt = "What is BENIGN network traffic? Describe its normal behavior, characteristics, and why it's important to differentiate it from malicious traffic."
    elif label in ["DOS", "DDOS", "DOS HULK", "DOS SLOWLORIS", "DOS GOLDENEYE"]:
        prompt = f"Explain in depth the '{label}' attack. Cover how it works, its method, impact on infrastructure, and defenses like firewalls or rate-limiting."
    elif label in ["XSS", "BRUTE FORCE", "PORTSCAN", "BOT", "INFILTRATION", "PORT SCAN"]:
        prompt = f"You're a cybersecurity analyst. Describe how a '{label}' attack is executed, its target, technical behavior, risks, and how it's mitigated."
    else:
        prompt = f"Explain the network threat '{label}' with context on vectors, systems affected, and common protective measures."
    
    # For less detailed reports, simplify the prompt
    if not detailed:
        prompt = f"Briefly explain the network threat '{label}' in 2-3 sentences."
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            inputs["input_ids"],
            max_length=400 if detailed else 150,
            do_sample=True,
            temperature=0.7,
            top_k=40,
            top_p=0.9
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating LLM insight: {e}")
        return f"LLM service unavailable: {str(e)}"

def save_report_to_db(attack_type, manual_summary, llm_insight):
    try:
        conn = sqlite3.connect("ids_reports.db")
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO reports (attack_type, timestamp, manual_summary, llm_insight)
            VALUES (?, ?, ?, ?)
        ''', (attack_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), manual_summary, llm_insight))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Failed to save report: {e}")

def generate_report(label, report_type="standard", include_tips=True):
    """Generate a comprehensive security report"""
    try:
        if not label or label.strip() == "":
            return "⚠️ Please enter a valid attack type."
        
        decoded = decode_label(label).upper()
        source = attack_source_map.get(decoded, "ml")
        model_used = "Machine Learning Model" if source == "ml" else "Federated-Reinforcement Learning Model"
        explanation = attack_explanations.get(decoded, "No explanation available.")
        
        # Handle concise summary
        if report_type == "concise":
            insight = generate_llm_insight(decoded, detailed=False)
            save_report_to_db(decoded, explanation, insight)

            return f"""
🛡️ **Intrusion Detection Report - Executive Summary**
───────────────────────────────────────
✅ **Alert:** {decoded} traffic detected  
🧠 **Source:** {model_used}

📘 **Summary:**  
{explanation}

🤖 **LLM Insight:**  
{insight}

📅 Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

        # Handle standard and detailed reports
        insight = generate_llm_insight(decoded, detailed=True)
        save_report_to_db(decoded, explanation, insight)

        report = f"""
🛡️ **Intrusion Detection System Report**
───────────────────────────────────────
✅ **Predicted Attack Type:** {decoded}  
🧠 **Source Model Used:** {model_used}  

📘 **Manual Summary:**  
{explanation}

🤖 **LLM Insight:**  
{insight}
"""

        if include_tips and report_type == "detailed":
            tips_prompt = f"List 3-5 specific mitigation steps for a '{decoded}' attack."
            inputs = tokenizer(tips_prompt, return_tensors="pt").to(device)
            tips_ids = model.generate(
                inputs["input_ids"],
                max_length=250,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
            tips = tokenizer.decode(tips_ids[0], skip_special_tokens=True)
            report += f"""

🛠️ **Recommended Mitigation Steps:**  
{tips}
"""

        report += f"\n📅 Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        return report

    except Exception as e:
        print(f"Error generating report: {e}")
        return f"⚠️ Error generating report: {str(e)}"

# Fix for the load_reports_incrementally function
def load_reports_incrementally(index=0, cache=None, include_benign=False, report_type="standard"):
    """Load batch reports from CSV files"""
    if cache is None:
        cache = []
    try:
        # Read available CSV files
        available_files = []
        for filename in ["llm_input_data.csv", "llm_input_data_rf.csv", "llm_input_data_lstm.csv"]:
            if os.path.exists(filename):
                available_files.append(filename)
        
        if not available_files:
            return "⚠️ No input data files found. Please ensure CSV files are present.", index, []
            
        dataframes = []
        for file in available_files:
            try:
                df = pd.read_csv(file, dtype=str)
                dataframes.append(df)
            except Exception as e:
                print(f"Warning: Could not read {file}: {e}")
        
        if not dataframes:
            return "❗ Could not read any CSV files. Check file format and permissions.", index, []
            
        df_combined = pd.concat(dataframes, ignore_index=True)
    except Exception as e:
        return f"❗ CSV Read Error: {e}", index, []

    # Process the labels
    all_labels = [decode_label(lbl) for lbl in df_combined['label'].tolist()]
    if not include_benign:
        all_labels = [lbl for lbl in all_labels if lbl.upper() != "BENIGN"]

    # Filter out invalid entries
    clean_labels = [lbl for lbl in all_labels if isinstance(lbl, str) and lbl.strip()]
    total = len(clean_labels)
    
    if total == 0:
        return "No valid labels found in the data files.", index, []

    # Get batch for current index
    batch = clean_labels[index:index+10]

    # Generate reports for batch
    results = []
    for lbl in batch:
        try:
            # Generate report with same parameters as single reports
            report = generate_report(lbl, report_type=report_type, include_tips=(report_type=="detailed"))
            results.append(report)
            cache.append(lbl)
        except Exception as e:
            results.append(f"⚠️ Error generating report for {lbl}: {str(e)}")

    next_index = index + len(batch)
    
    # Progress indicator
    progress = min(next_index, total) / total * 100
    progress_bar = f"[{'=' * int(progress // 10)}{' ' * (10 - int(progress // 10))}] {progress:.1f}%"
    
    end_msg = f"\n\n📊 Progress: {progress_bar}"
    end_msg += f"\n📄 Showing {min(next_index, total)} of {total} reports."
    
    if next_index >= total:
        end_msg += "\n✅ All reports loaded."

    return "\n\n".join(results) + end_msg, next_index, cache

def safe_pdf_config():
    """Get safe PDF configuration"""
    path = shutil.which('wkhtmltopdf')
    if not path:
        # Common installation paths
        potential_paths = [
            r"C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe",
            r"/usr/bin/wkhtmltopdf",
            r"/usr/local/bin/wkhtmltopdf"
        ]
        for p in potential_paths:
            if os.path.exists(p):
                path = p
                break
    
    if not path:
        print("Warning: wkhtmltopdf not found in PATH or common locations")
        
    return pdfkit.configuration(wkhtmltopdf=path)

def generate_pdf(content, title="IDS Report"):
    """Generate formatted PDF"""
    try:
        config = safe_pdf_config()
        
        # Create a unique filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"ids_report_{sanitize_filename(title)}_{timestamp}.pdf"
        output_path = str(CACHE_DIR / filename)
        
        # CSS for styling
        css = """
        body {
            font-family: 'Arial', sans-serif;
            background: #121212;
            color: #E0E0E0;
            padding: 30px;
            line-height: 1.6;
        }
        h1, h2 {
            color: #6ab7ff;
            border-bottom: 1px solid #6ab7ff;
            padding-bottom: 10px;
        }
        hr {
            border: 1px solid #6ab7ff;
        }
        pre {
            font-size: 14px;
            white-space: pre-wrap;
            background: #1c1c1c;
            padding: 15px;
            border-radius: 5px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .footer {
            text-align: center;
            font-size: 10px;
            margin-top: 40px;
            border-top: 1px solid #555;
            padding-top: 10px;
        }
        """
        
        # Convert Markdown formatting to HTML
        content_html = content.replace("**", "<strong>", 1)
        content_html = content_html.replace("**", "</strong>", 1)
        content_html = content_html.replace("**", "<strong>")
        content_html = content_html.replace("**", "</strong>")
        
        html = f"""
        <html>
            <head>
                <meta charset="utf-8">
                <title>{title}</title>
                <style>{css}</style>
            </head>
            <body>
                <div class="header">
                    <h1>🚀 Next-Gen IDS Secure Report</h1>
                </div>
                <pre>{content_html}</pre>
                <div class="footer">
                    Confidential © 2025 Next-Gen IDS - Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}
                </div>
            </body>
        </html>
        """
        
        # Generate PDF
        pdfkit.from_string(html, output_path, configuration=config, options={
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-right': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm',
            'encoding': 'UTF-8',
            'no-outline': None
        })
        
        return output_path
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None

# Fix for batch PDF generation
def generate_batch_pdf(content):
    """Generate PDF for batch reports"""
    try:
        if not content or content.strip() == "":
            return None
            
        # Use a more descriptive title
        return generate_pdf(content, title="IDS_Batch_Security_Report")
    except Exception as e:
        print(f"Error in batch PDF generation: {e}")
        return None

# ========== Custom Theme ==========
cybersecurity_theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "system-ui"],
).set(
    # Light and dark mode compatibility
    body_background_fill="#121212",
    block_background_fill="#1c1c1c",
    block_label_background_fill="#1c1c1c",
    block_title_text_color="#6ab7ff",
    body_text_color="#E0E0E0",
    border_color_primary="#383838",
    button_primary_background_fill="#1F6FEB",
    button_primary_background_fill_hover="#2977FF",
    button_primary_text_color="white",
    button_secondary_background_fill="#2D2D2D",
    button_secondary_background_fill_hover="#3D3D3D",
    button_secondary_text_color="#E0E0E0",
    checkbox_background_color="#2D2D2D",
    checkbox_background_color_selected="#1F6FEB",
    checkbox_border_color="#4D4D4D",
    checkbox_border_color_focus="#6ab7ff",
    checkbox_border_color_hover="#6D6D6D",
    checkbox_border_color_selected="#1F6FEB",
    checkbox_label_background_fill="#1c1c1c",
    input_background_fill="#242424",
    input_border_color="#383838",
    input_border_color_focus="#6ab7ff",
    input_placeholder_color="#909090",
    input_shadow="#1c1c1c",
    slider_color="#1F6FEB",
    slider_color_dark="#1F6FEB",
    stat_background_fill="#2D2D2D",
)

# ========== Custom CSS ==========
custom_css = """
/* Tooltips and interactive elements */
[data-testid="tooltip"] {
    background-color: #1F6FEB !important;
    color: white !important;
}

/* Code highlighting */
.gr-code textarea {
    background-color: #1c1c1c !important;
    color: #00FF00 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Button styles */
.gr-button {
    font-weight: bold !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
}
.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
}

/* Gradient accent elements */
.header-accent {
    background: linear-gradient(90deg, #1F6FEB, #6ab7ff) !important;
    height: 3px !important;
    margin-bottom: 20px !important;
}

/* Toast notifications */
#toast {
    visibility: hidden;
    min-width: 250px;
    background-color: #1F6FEB;
    color: white;
    text-align: center;
    border-radius: 8px;
    padding: 16px;
    position: fixed;
    z-index: 9999;
    bottom: 30px;
    right: 30px;
    font-size: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
#toast.show {
    visibility: visible;
    animation: fadein 0.5s, fadeout 0.5s 2.5s;
}
@keyframes fadein {
    from {bottom: 0; opacity: 0;}
    to {bottom: 30px; opacity: 1;}
}
@keyframes fadeout {
    from {bottom: 30px; opacity: 1;}
    to {bottom: 0; opacity: 0;}
}

/* Card styles */
.custom-card {
    border: 1px solid #383838 !important;
    border-radius: 10px !important;
    padding: 15px !important;
    margin-bottom: 15px !important;
    background-color: #1c1c1c !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    transition: all 0.3s ease !important;
}
.custom-card:hover {
    box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
    border-color: #6ab7ff !important;
}

/* Tabs styling */
.tabs {
    border-bottom: 2px solid #383838 !important;
}
.tab-selected {
    border-bottom: 2px solid #1F6FEB !important;
    font-weight: bold !important;
}

/* Progress bar */
.progress-container {
    width: 100%;
    background-color: #2D2D2D;
    border-radius: 4px;
    margin: 10px 0;
}
.progress-bar {
    height: 8px;
    background: linear-gradient(90deg, #1F6FEB, #6ab7ff);
    border-radius: 4px;
    transition: width 0.3s ease;
}

/* Dropdown enhancements */
.gr-dropdown {
    background-color: #242424 !important;
}
.gr-dropdown:focus {
    border-color: #6ab7ff !important;
}

/* Modal styling */
.modal {
    background-color: #1c1c1c !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5) !important;
}
.modal-header {
    border-bottom: 1px solid #383838 !important;
}
"""

# ========== JavaScript Functions ==========
js_functions = """
<script>
// Toast notification handler
function showToast(msg) {
    var t = document.getElementById("toast");
    if (!t) {
        t = document.createElement("div");
        t.id = "toast";
        document.body.appendChild(t);
    }
    t.innerText = msg;
    t.className = "show";
    setTimeout(function(){ t.className = t.className.replace("show", ""); }, 3000);
}

// Copy report to clipboard
function copyReport() {
    const textarea = document.querySelector('textarea');
    if (textarea) {
        navigator.clipboard.writeText(textarea.value)
            .then(() => showToast('📋 Report copied to clipboard!'))
            .catch(err => showToast('❌ Failed to copy: ' + err));
    }
}

// Auto-download PDF when generated
const observer = new MutationObserver(function(mutations) {
    for (let m of mutations) {
        if (m.type === 'childList' && m.addedNodes.length > 0) {
            let link = document.querySelector('a[href$=".pdf"]');
            if (link) {
                link.click();
                showToast('✅ PDF ready and downloading...');
            }
        }
    }
});
observer.observe(document.body, {childList: true, subtree: true});

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+Enter to generate report
    if (e.ctrlKey && e.key === 'Enter') {
        const generateBtn = document.querySelector('button[aria-label="🛡️ Generate Report"]');
        if (generateBtn) generateBtn.click();
    }
    // Ctrl+S to save PDF
    if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        const downloadBtn = document.querySelector('button[aria-label="⬇️ Download PDF"]');
        if (downloadBtn) downloadBtn.click();
    }
    // Ctrl+C to copy (custom handler)
    if (e.ctrlKey && e.key === 'c' && window.getSelection().toString() === '') {
        copyReport();
    }
});

// Theme toggle (prepared for future use)
function toggleTheme() {
    document.body.classList.toggle('light-theme');
    localStorage.setItem('theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
}

// Security animation
function securityAnimation() {
    const container = document.createElement('div');
    container.style.position = 'fixed';
    container.style.top = '0';
    container.style.left = '0';
    container.style.width = '100%';
    container.style.height = '100%';
    container.style.pointerEvents = 'none';
    container.style.zIndex = '1000';
    document.body.appendChild(container);
    
    for (let i = 0; i < 100; i++) {
        const dot = document.createElement('div');
        dot.style.position = 'absolute';
        dot.style.width = Math.random() * 5 + 'px';
        dot.style.height = dot.style.width;
        dot.style.backgroundColor = 'rgba(31, 111, 235, 0.7)';
        dot.style.borderRadius = '50%';
        dot.style.top = Math.random() * 100 + '%';
        dot.style.left = Math.random() * 100 + '%';
        
        container.appendChild(dot);
        
        setTimeout(() => {
            dot.style.transition = 'all 2s ease-in-out';
            dot.style.transform = 'translateY(' + (Math.random() * 300 - 150) + 'px)';
            dot.style.opacity = '0';
        }, Math.random() * 1000);
    }
    
    setTimeout(() => container.remove(), 3000);
}
</script>
<div id="toast"></div>
"""

# ========== Gradio Interface ==========
with gr.Blocks(
    theme=cybersecurity_theme,
    css=custom_css,
    title="Next-Gen IDS Assistant"
) as app:

    gr.HTML(js_functions)

    gr.HTML("""
    <div style="text-align:center; margin-bottom: 20px;">
        <h1 style="color:#6ab7ff; margin-bottom:0;">🛡️ Next-Gen IDS Assistant</h1>
        <p style="color:#E0E0E0; font-size:1.1em;">Secure | Intelligent | Real-Time Detection</p>
        <div class="header-accent"></div>
    </div>
    """)

    with gr.Tabs(elem_classes="tabs") as tabs:
        with gr.Tab("🔍 Single Report", elem_classes="tab-selected"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""<div class="custom-card">
                        <h3 style="color:#6ab7ff; margin-top:0;">📊 Attack Classification</h3>
                        <p>Analyze a specific attack type and generate a detailed security report with AI insights.</p>
                    </div>""")
                    
                    attack_dropdown = gr.Dropdown(
                        choices=ALL_ATTACK_TYPES,
                        label="Choose Attack Type",
                        info="Select from known attack classifications",
                        interactive=True
                    )
                    
                    label_input = gr.Textbox(
                        label="Or Enter Custom Attack Class",
                        placeholder="e.g., Bot, DDoS, Infiltration",
                        info="Custom inputs will be normalized if possible"
                    )
                    
                    report_type = gr.Radio(
                        choices=["concise", "standard", "detailed"],
                        value="standard",
                        label="Report Detail Level",
                        info="Choose how detailed the generated report should be"
                    )
                    
                    with gr.Accordion("Advanced Options", open=False):
                        include_tips = gr.Checkbox(
                            label="Include Mitigation Tips",
                            value=True,
                            info="Add mitigation recommendations to detailed reports"
                        )
                    
                    with gr.Row():
                        generate_btn = gr.Button("🛡️ Generate Report", variant="primary")
                        download_btn = gr.Button("⬇️ Download PDF")
                        copy_btn = gr.HTML("""<button onclick='copyReport()' 
                                          style='background:#2D2D2D; color:white; padding:10px; 
                                          border:none; border-radius:4px; cursor:pointer;'>
                                          📋 Copy Report</button>""")

                with gr.Column(scale=2):
                    report_output = gr.Markdown(label="Generated Security Report", elem_id="report-output", value="")
                    single_pdf_output = gr.File(label="Generated PDF", visible=False)

            # Connect report generation to multiple input sources
            def generate_combined_report(dropdown_val, text_val, report_type, include_tips):
                # Use dropdown if provided, otherwise use text input
                attack_val = dropdown_val if dropdown_val else text_val
                if not attack_val or attack_val.strip() == "":
                    return "⚠️ Please select or enter an attack type to analyze."
                return generate_report(attack_val, report_type, include_tips)
                
            generate_btn.click(
                fn=generate_combined_report,
                inputs=[attack_dropdown, label_input, report_type, include_tips],
                outputs=report_output
            )
            
            # Quick submit on enter
            label_input.submit(
                fn=generate_combined_report,
                inputs=[attack_dropdown, label_input, report_type, include_tips],
                outputs=report_output
            )
            
            # PDF generation 
            download_btn.click(
                fn=generate_pdf,
                inputs=report_output,
                outputs=single_pdf_output
            )

        with gr.Tab("🧾 Batch Reports"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""<div class="custom-card">
                        <h3 style="color:#6ab7ff; margin-top:0;">🔄 Batch Analysis</h3>
                        <p>Process multiple attack classifications from your data files to generate comprehensive reports.</p>
                    </div>""")
                    
                    include_benign_checkbox = gr.Checkbox(
                        label="✅ Include BENIGN Traffic",
                        value=False,
                        info="Include normal traffic patterns in analysis"
                    )
                    
                    batch_report_type = gr.Radio(
                        choices=["concise", "standard", "detailed"],
                        value="standard",
                        label="Batch Report Detail Level"
                    )
                    
                    with gr.Row():
                        load_btn = gr.Button("➕ Load Next 10 Reports", variant="primary")
                        batch_download_btn = gr.Button("⬇️ Download Batch PDF")
                        reset_btn = gr.Button("🔄 Reset", variant="secondary")
                        
                    gr.HTML("""<div class="custom-card">
                        <h3 style="color:#6ab7ff; margin-top:0;">📁 Data Source</h3>
                        <p>Reports are generated from these files when available:</p>
                        <ul>
                            <li>llm_input_data.csv</li>
                            <li>llm_input_data_rf.csv</li>
                            <li>llm_input_data_lstm.csv</li>
                        </ul>
                    </div>""")

                with gr.Column(scale=2):
                    batch_output = gr.Markdown(
                        label="Batch Reports",
                        elem_id="batch-output",
                        value="Click 'Load Next 10 Reports' to begin batch processing."
                    )
                    report_state = gr.State(0)
                    report_cache = gr.State([])
                    batch_pdf_output = gr.File(label="Generated Batch PDF", visible=False)
            
            # Connect batch report generation
            def load_batch(index, cache, include_benign, report_type):
                return load_reports_incrementally(index, cache, include_benign, report_type)
                
            load_btn.click(
                fn=load_batch,
                inputs=[report_state, report_cache, include_benign_checkbox, batch_report_type],
                outputs=[batch_output, report_state, report_cache]
            )
            
            batch_download_btn.click(
                fn=generate_batch_pdf,
                inputs=batch_output,
                outputs=batch_pdf_output
            )
            
            reset_btn.click(
                fn=lambda: (0, [], "Click 'Load Next 10 Reports' to begin batch processing."),
                inputs=None,
                outputs=[report_state, report_cache, batch_output]
            )

        with gr.Tab("📚 Knowledge Base"):
            with gr.Accordion("Attack Types Reference", open=True):
                reference_md = "## 🛡️ Common Network Attack Types\n\n"
                for attack, desc in sorted(attack_explanations.items()):
                    reference_md += f"### {attack}\n{desc}\n\n"
                
                gr.Markdown(reference_md)
                
            with gr.Accordion("Detection Methods", open=False):
                gr.Markdown("""
                ## 🔍 Detection Methodologies
                
                ### Machine Learning Models
                - **Feature-based Detection**: Analyzes packet metadata and flow characteristics
                - **Anomaly Detection**: Identifies deviations from normal traffic patterns
                - **Signature Detection**: Matches traffic against known attack signatures
                
                ### Federated Learning Models
                - **Distributed Detection**: Leverages models trained across multiple nodes
                - **Privacy-preserving**: Trains without sharing raw data
                - **Adaptive Learning**: Continually updates with new attack patterns
                """)

        with gr.Tab("⚙️ Settings"):
            gr.Markdown("## Application Settings")
            
            with gr.Row():
                with gr.Column():
                    # Placeholder for future settings
                    gr.Checkbox(label="Dark Mode (Default)", value=True, interactive=False,
                               info="Light mode coming in future update")
                    gr.Checkbox(label="Auto-download PDF Reports", value=True, interactive=False,
                               info="Automatically trigger download when PDF is generated")
                    
                with gr.Column():
                    gr.Checkbox(label="Use GPU Acceleration", value=torch.cuda.is_available(), interactive=False,
                               info="Hardware acceleration status (detected automatically)")
                    cache_location = gr.Textbox(label="Cache Directory", value=str(CACHE_DIR), interactive=False)
            
            with gr.Accordion("Advanced Configuration", open=False):
                gr.Code(
                    """# Example configuration (read-only)
{
  "model": "google/flan-t5-base",
  "device": "cuda" if torch.cuda.is_available() else "cpu",
  "temperature": 0.7,
  "max_length": 400,
  "cache_dir": "./cache",
  "seed": 42
}""", language="json")
            
            with gr.Row():
                clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary")
                
            clear_cache_btn.click(
                fn=lambda: (shutil.rmtree(CACHE_DIR, ignore_errors=True), 
                           CACHE_DIR.mkdir(exist_ok=True),
                           "✅ Cache cleared successfully!"),
                inputs=None,
                outputs=gr.Textbox(label="Status")
            )
    
    # Footer
    gr.HTML("""
    <div style="text-align:center; margin-top:30px; padding-top:15px; border-top:1px solid #383838;">
        <p style="color:#909090; font-size:0.9em;">© 2025 Next-Gen IDS Assistant | v1.2.0 | Built with Gradio and T5</p>
    </div>
    """)
    
# Launch the app
if __name__ == "__main__":
    app.queue(max_size=10).launch(debug=True, share=False)