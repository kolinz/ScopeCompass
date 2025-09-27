# -*- coding: utf-8 -*-
"""
このツールは、クラウドサービスの責任共有モデル可視化（CSRMV）ツールです。
CSRMVは、Cloud Services Shared Responsibility Model Visualization の略称です。
すべてのセルを実行してください。最後のセルにツールが表示されます。
Google ColabなどのJupyter Notebook / Lab環境にこのコードを貼り付けて動かすことが最も手軽です。
CSVダウンロードを使う場合は、「グラフ更新」を押してからにしてください。
"""
# ==============================================================================
# ライブラリのインストール
# ==============================================================================
!!pip install -q ipywidgets matplotlib japanize-matplotlib numpy pandas pillow

# ==============================================================================
# ライブラリのインポート
# ==============================================================================
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import japanize_matplotlib
import numpy as np
import io
import pandas as pd
import base64
from IPython.display import display, clear_output
from PIL import Image
from urllib.parse import quote
from google.colab import output

# ==============================================================================
# 定数と初期値の定義
# ==============================================================================
LAYERS = [
    "データ & アクセス", "アプリケーション", "ランタイム/コンテナ",
    "ミドルウェア", "OS", "仮想化基盤",
    "物理サーバー", "物理ストレージ", "ネットワーク機器"
]
MODELS = ["On-Premise", "Private Cloud", "Public IaaS", "Public PaaS", "Public SaaS"]
PARTIES = {"End User": "#FFEB3B", "SIer": "#1976D2", "CSP": "#757575"}

_initial_vals_two_party = {
    "On-Premise": [[0, 100], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0]],
    "Private Cloud": [[0, 100], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0]],
    "Public IaaS": [[0, 100], [100, 0], [100, 0], [100, 0], [100, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    "Public PaaS": [[0, 100], [100, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    "Public SaaS": [[0, 100], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
}
INITIAL_VALUES = {
    model: [[s, u, 100 - (s + u)] for s, u in values]
    for model, values in _initial_vals_two_party.items()
}

# ==============================================================================
# コアロジック関数
# ==============================================================================
def draw_chart_for_ipywidgets(sier_values, user_values, csp_values, active_models):
    num_layers, num_active_models = sier_values.shape
    fig_width = max(10, num_active_models * 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 7), dpi=100)

    for i in range(num_layers):
        for j in range(num_active_models):
            total = sier_values[i, j] + user_values[i, j] + csp_values[i, j]
            if total == 0: continue
            
            percentages_dict = {"End User": user_values[i, j] / total, "SIer": sier_values[i, j] / total, "CSP": csp_values[i, j] / total}
            stack_order = ["End User", "SIer", "CSP"]
            current_y = i
            for party in stack_order:
                height = percentages_dict[party]
                if height > 0:
                    ax.add_patch(patches.Rectangle((j, current_y), 1, height, facecolor=PARTIES[party], edgecolor='white', linewidth=0.5))
                current_y += height

    ax.set_xlim(-1.2, num_active_models)
    ax.set_ylim(num_layers, -0.5)
    ax.axis('off')

    for j, model in enumerate(active_models):
        ax.text(j + 0.5, -0.2, model, ha='center', va='bottom', fontsize=12, weight='bold')
    for i, layer in enumerate(LAYERS):
        ax.text(-0.1, i + 0.5, layer, ha='right', va='center', fontsize=10)

    legend_labels_ordered = {"End User": "End User", "SIer": "SIer", "CSP": "CSP (クラウド事業者)"}
    legend_elements = [patches.Patch(facecolor=PARTIES[name], label=label) for name, label in legend_labels_ordered.items()]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=12)
    
    plt.tight_layout(pad=1.0)
    plt.show()

# 選択されたモデルのリストを引数で受け取る
def create_download_link(input_widgets, selected_models):
    """選択されたモデルの入力値からCSVを生成し、ダウンロードリンクのHTMLを返す"""
    data = []
    # 全モデル(MODELS)ではなく、選択されたモデル(selected_models)でループ
    for model in selected_models:
        for layer in LAYERS:
            u_val = input_widgets[model][layer]['End User'].value
            s_val = input_widgets[model][layer]['SIer'].value
            c_val = input_widgets[model][layer]['CSP'].value
            data.append([model, layer, u_val, s_val, c_val])
    
    df = pd.DataFrame(data, columns=['サービスモデル', '責任範囲', 'End User', 'SIer', 'CSP'])
    
    csv_str = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv_str.encode()).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="responsibility_model.csv">現在の選択範囲をCSVでダウンロード</a>'
    return href

# ==============================================================================
# ipywidgets UIの構築とイベント処理
# ==============================================================================

# --- UIウィジェットの作成 ---
title = widgets.HTML("<h2>クラウド責任共有モデル 可視化ツール</h2>")
description = widgets.HTML("<p>表示するモデルを選択し、各タブで数値を入力後、「グラフを更新」ボタンを押してください。</p>")
model_selector = widgets.SelectMultiple(
    options=MODELS,
    value=MODELS,
    description='表示モデル:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='95%')
)
update_button = widgets.Button(description="グラフを更新", button_style='primary')
status_display = widgets.HTML("")
download_link_display = widgets.HTML(value="")

tab_children = []
input_widgets = {}
for model in MODELS:
    layer_inputs = []
    input_widgets[model] = {}
    for layer in LAYERS:
        layer_index = LAYERS.index(layer)
        s_val, u_val, c_val = INITIAL_VALUES[model][layer_index]
        u_input = widgets.IntText(value=u_val, description="End User %", style={'description_width': 'initial'})
        s_input = widgets.IntText(value=s_val, description="SIer %", style={'description_width': 'initial'})
        c_input = widgets.IntText(value=c_val, description="CSP %", style={'description_width': 'initial'})
        input_widgets[model][layer] = {'End User': u_input, 'SIer': s_input, 'CSP': c_input}
        layer_group = widgets.VBox([widgets.HTML(f"<b>{layer}</b>"), u_input, s_input, c_input])
        layer_inputs.append(layer_group)
    tab_children.append(widgets.VBox(layer_inputs))

input_tabs = widgets.Tab()
input_tabs.children = tab_children
for i, model in enumerate(MODELS):
    input_tabs.set_title(i, model)

chart_output = widgets.Output()

# --- イベントハンドラ関数の定義 ---
def on_update_button_clicked(b):
    chart_output.clear_output(wait=True)
    
    selected_models = list(model_selector.value)
    if not selected_models:
        status_display.value = "<p style='color:red;'>❌ エラー: 表示するサービスモデルを1つ以上選択してください。</p>"
        download_link_display.value = "" # エラー時はリンクを消去
        return
    selected_models.sort(key=MODELS.index)

    for model in selected_models:
        for layer in LAYERS:
            u_val = input_widgets[model][layer]['End User'].value
            s_val = input_widgets[model][layer]['SIer'].value
            c_val = input_widgets[model][layer]['CSP'].value
            total = u_val + s_val + c_val
            if total != 100:
                status_display.value = f"<p style='color:red;'>❌ エラー: 「{model}」の「{layer}」の合計が {total} です。100に修正してください。</p>"
                download_link_display.value = "" # エラー時はリンクを消去
                return
    
    selected_indices = [MODELS.index(model) for model in selected_models]
    num_layers = len(LAYERS)
    sier_to_draw = np.zeros((num_layers, len(selected_models)), dtype=int)
    user_to_draw = np.zeros((num_layers, len(selected_models)), dtype=int)
    csp_to_draw = np.zeros((num_layers, len(selected_models)), dtype=int)
    for col_idx, model in enumerate(selected_models):
        for row_idx, layer in enumerate(LAYERS):
            user_to_draw[row_idx, col_idx] = input_widgets[model][layer]['End User'].value
            sier_to_draw[row_idx, col_idx] = input_widgets[model][layer]['SIer'].value
            csp_to_draw[row_idx, col_idx] = input_widgets[model][layer]['CSP'].value

    with chart_output:
        draw_chart_for_ipywidgets(sier_to_draw, user_to_draw, csp_to_draw, selected_models)
    
    status_display.value = f"<p style='color:green;'>✅ グラフを正常に更新しました。（{', '.join(selected_models)}）</p>"
    download_link_display.value = create_download_link(input_widgets, selected_models)

# --- イベントの接続 ---
update_button.on_click(on_update_button_clicked)

# --- UI全体のレイアウト ---
control_panel = widgets.VBox([
    title, 
    description,
    model_selector,
    update_button,
    status_display,
    download_link_display, 
    widgets.HTML("<h3>調整パネル</h3>"),
    input_tabs
])
app_layout = widgets.HBox([
    control_panel,
    widgets.VBox([widgets.HTML("<h3>責任分担図</h3>"), chart_output])
])

# アプリケーションを表示し、初期描画を実行する
display(app_layout)
on_update_button_clicked(None)
output.eval_js("new Promise(resolve => setTimeout(() => {document.querySelector('#output-area').scrollIntoView({ behavior: 'smooth', block: 'start' }); resolve();}, 200))")
