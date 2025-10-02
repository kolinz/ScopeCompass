# -*- coding: utf-8 -*-
"""
このツールは、クラウドサービスの責任共有モデル可視化（CSRMV）ツールです。
CSRMVは、Cloud Services Shared Responsibility Model Visualization の略称です。
Google Colab以外のJupyter Notebook/Lab環境にこのコードを貼り付けて動かしてください。watsonx.ai Studio上のJupyter Notebook向けにGradioを使っています。
CSVダウンロードを使う場合は、「グラフを更新」を押してからにしてください。
ライブラリ関連エラーは、カーネルの再起動で解決することがあります。
"""
# ==============================================================================
# 1. ライブラリのインストールとインポート
# ==============================================================================
# Gradioをインストール（まだインストールされていない場合）
!pip install --upgrade -q gradio matplotlib japanize-matplotlib numpy pandas pillow

import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import japanize_matplotlib
import numpy as np
import pandas as pd
import textwrap
import tempfile # CSVダウンロードのために追加

# ==============================================================================
# 2. 定数と初期値の定義 (元のコードから変更なし)
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
INITIAL_VALUES = {model: [[s, u, 100 - (s + u)] for s, u in values] for model, values in _initial_vals_two_party.items()}

# ==============================================================================
# 3. コアロジック関数 (元のコードから変更なし)
# ==============================================================================
def create_chart_figure(sier_values, user_values, csp_values, active_models, org_names):
    """グラフのFigureオブジェクトを生成して返す（表示はしない）"""
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

    ax.set_xlim(-1.2, num_active_models); ax.set_ylim(num_layers, -0.5); ax.axis('off')
    for j, model in enumerate(active_models): ax.text(j + 0.5, -0.2, model, ha='center', va='bottom', fontsize=12, weight='bold')
    for i, layer in enumerate(LAYERS): ax.text(-0.1, i + 0.5, layer, ha='right', va='center', fontsize=10)
    
    wrapped_org_names = {role: textwrap.fill(name, width=20) for role, name in org_names.items()}
    legend_labels_ordered = {
        "End User": f"End User ({wrapped_org_names['End User']})", "SIer": f"SIer ({wrapped_org_names['SIer']})", "CSP": f"CSP ({wrapped_org_names['CSP']})"
    }
    legend_elements = [patches.Patch(facecolor=PARTIES[name], label=label) for name, label in legend_labels_ordered.items()]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False, fontsize=10)
    plt.tight_layout(pad=1.0)
    fig.subplots_adjust(bottom=0.1)
    
    return fig

# ==============================================================================
# 4. Gradio UIの構築とイベント処理
# ==============================================================================
def update_chart_and_data(*args):
    """UIの入力値を受け取り、グラフとCSVファイルを更新するGradio用のメイン関数"""
    # argsから各入力値を取り出す
    selected_models = args[0]
    org_user, org_sier, org_csp = args[1], args[2], args[3]
    # 残りのargsはすべて数値入力フィールド
    value_inputs = args[4:]
    
    # --- バリデーション ---
    if not selected_models:
        error_msg = "❌ エラー: 表示するサービスモデルを1つ以上選択してください。"
        # エラー時は現在のプロットをクリアし、メッセージを返す
        return None, error_msg, None

    # Gradioの入力値を元のコードのデータ構造に変換
    input_data = {}
    value_idx = 0
    for model in MODELS:
        input_data[model] = {}
        for layer in LAYERS:
            input_data[model][layer] = {
                'End User': value_inputs[value_idx],
                'SIer': value_inputs[value_idx + 1],
                'CSP': value_inputs[value_idx + 2],
                'Note': value_inputs[value_idx + 3]
            }
            value_idx += 4

    for model in selected_models:
        for layer in LAYERS:
            total = sum([input_data[model][layer][p] for p in ['End User', 'SIer', 'CSP']])
            if total != 100:
                error_msg = f"❌ エラー: 「{model}」の「{layer}」の合計が {int(total)}% です。100%に修正してください。"
                return None, error_msg, None

    # --- グラフ描画データ作成 ---
    selected_models.sort(key=MODELS.index)
    num_layers = len(LAYERS)
    sier, user, csp = [np.zeros((num_layers, len(selected_models)), dtype=int) for _ in range(3)]
    for col_idx, model in enumerate(selected_models):
        for row_idx, layer in enumerate(LAYERS):
            user[row_idx, col_idx] = input_data[model][layer]['End User']
            sier[row_idx, col_idx] = input_data[model][layer]['SIer']
            csp[row_idx, col_idx] = input_data[model][layer]['CSP']
    
    org_names = {'End User': org_user, 'SIer': org_sier, 'CSP': org_csp}
    fig = create_chart_figure(sier, user, csp, selected_models, org_names)
    status_msg = f"✅ グラフを正常に更新しました。（{', '.join(selected_models)}）"
    
    # --- CSVファイル作成 ---
    csv_data = []
    for model in selected_models:
        for layer in LAYERS:
            u_val = input_data[model][layer]['End User']
            s_val = input_data[model][layer]['SIer']
            c_val = input_data[model][layer]['CSP']
            note = input_data[model][layer]['Note']
            csv_data.append([model, layer, u_val, s_val, c_val, note])
    
    df = pd.DataFrame(csv_data, columns=['サービスモデル', 'ITレイヤー', 'End User', 'SIer', 'CSP', '注釈'])
    
    # 一時ファイルにCSVを保存してそのパスを返す
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv', encoding='utf-8-sig') as f:
        df.to_csv(f.name, index=False)
        csv_filepath = f.name
        
    return fig, status_msg, csv_filepath

# GradioのUIレイアウトを定義
with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as demo:
    gr.Markdown("## クラウド責任共有モデル 可視化ツール")
    gr.Markdown("役割別の組織名や表示モデルを選択し、各タブで数値を入力後、「グラフを更新」ボタンを押してください。")
    
    # UI要素を格納するリスト
    inputs = []

    with gr.Row():
        # --- 左側の操作パネル ---
        with gr.Column(scale=1):
            model_selector = gr.CheckboxGroup(choices=MODELS, value=MODELS, label="表示モデル")
            inputs.append(model_selector)
            
            update_button = gr.Button("グラフを更新", variant="primary")
            status_display = gr.Markdown("")
            
            with gr.Accordion("役割別の組織名", open=True):
                org_user_input = gr.Textbox(value="顧客企業名", label="End User")
                org_sier_input = gr.Textbox(value="SIer企業名", label="SIer")
                org_csp_input = gr.Textbox(value="クラウド事業者名", label="CSP")
                inputs.extend([org_user_input, org_sier_input, org_csp_input])

            gr.Markdown("### 調整パネル")
            with gr.Tabs():
                all_value_inputs = []
                for model in MODELS:
                    with gr.TabItem(model):
                        for layer in LAYERS:
                            s_val, u_val, c_val = INITIAL_VALUES[model][LAYERS.index(layer)]
                            with gr.Accordion(layer, open=False):
                                u_input = gr.Number(value=u_val, label="End User %", precision=0)
                                s_input = gr.Number(value=s_val, label="SIer %", precision=0)
                                c_input = gr.Number(value=c_val, label="CSP %", precision=0)
                                note_input = gr.Textbox(label="補足事項", placeholder="補足事項があれば入力...")
                                all_value_inputs.extend([u_input, s_input, c_input, note_input])
                inputs.extend(all_value_inputs)

        # --- 右側の描画エリア ---
        with gr.Column(scale=2):
            gr.Markdown("### 責任分担図")
            chart_output = gr.Plot()
            csv_download_button = gr.File(label="CSVファイルをダウンロード", file_count="single")

    # ボタンがクリックされた時のイベントを設定
    update_button.click(
        fn=update_chart_and_data,
        inputs=inputs,
        outputs=[chart_output, status_display, csv_download_button]
    )
    
    # ページロード時に一度グラフを描画
    demo.load(
        fn=update_chart_and_data,
        inputs=inputs,
        outputs=[chart_output, status_display, csv_download_button]
    )

# Jupyter Notebook内でインライン表示
demo.launch(share=True, height=800)
