# -*- coding: utf-8 -*-
"""
このツールは、クラウドサービスの責任共有モデル可視化（CSRMV）ツールです。
CSRMVは、Cloud Services Shared Responsibility Model Visualization の略称です。
Google Colab（Google Colaboratory）に、このコードを貼り付け、セルを実行してください。
CSVダウンロードを使う場合は、「グラフを更新」を押してからにしてください。
"""
# ==============================================================================
# 1. ライブラリのインストールとインポート
# ==============================================================================
!pip install -q ipywidgets matplotlib japanize-matplotlib numpy pandas pillow gspread gspread-dataframe google-auth-oauthlib google-api-python-client

import ipywidgets as widgets
from IPython.display import display, clear_output, Javascript
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import japanize_matplotlib
import numpy as np
import io
from PIL import Image
import pandas as pd
import base64
from urllib.parse import quote
import textwrap
import datetime
import gspread
from google.auth import default
from gspread_dataframe import set_with_dataframe
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# ==============================================================================
# 2. 定数と初期値の定義
# ==============================================================================
# UIに表示するITレイヤーとサービスモデル
LAYERS = [
    "データ & アクセス", "アプリケーション", "ランタイム/コンテナ",
    "ミドルウェア", "OS", "仮想化基盤",
    "物理サーバー", "物理ストレージ", "ネットワーク機器"
]
MODELS = ["On-Premise", "Private Cloud", "Public IaaS", "Public PaaS", "Public SaaS"]
# 描画に使用する色
PARTIES = {"End User": "#FFEB3B", "SIer": "#1976D2", "CSP": "#757575"}

# アプリケーション起動時のデフォルト値
_initial_vals_two_party = {
    "On-Premise": [[0, 100], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0]],
    "Private Cloud": [[0, 100], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0]],
    "Public IaaS": [[0, 100], [100, 0], [100, 0], [100, 0], [100, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    "Public PaaS": [[0, 100], [100, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    "Public SaaS": [[0, 100], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
}
INITIAL_VALUES = {model: [[s, u, 100 - (s + u)] for s, u in values] for model, values in _initial_vals_two_party.items()}

# ==============================================================================
# 3. コアロジック関数
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

def get_dataframe_from_inputs(input_widgets, selected_models):
    """現在の入力値からpandas DataFrameを生成する"""
    data = []
    for model in selected_models:
        for layer in LAYERS:
            u_val = input_widgets[model][layer]['End User'].value
            s_val = input_widgets[model][layer]['SIer'].value
            c_val = input_widgets[model][layer]['CSP'].value
            note = input_widgets[model][layer]['Note'].value
            data.append([model, layer, u_val, s_val, c_val, note])
    return pd.DataFrame(data, columns=['サービスモデル', 'ITレイヤー', 'End User', 'SIer', 'CSP', '注釈'])

def validate_and_get_data(selected_models, input_widgets):
    """入力値を検証し、問題があればエラーメッセージを、なければ描画用データを返す"""
    if not selected_models:
        return None, "❌ エラー: 表示するサービスモデルを1つ以上選択してください。"
    for model in selected_models:
        for layer in LAYERS:
            total = sum([input_widgets[model][layer][p].value for p in ['End User', 'SIer', 'CSP']])
            if total != 100:
                return None, f"❌ エラー: 「{model}」の「{layer}」の合計が {total} です。100に修正してください。"
    
    num_layers = len(LAYERS)
    sier, user, csp = [np.zeros((num_layers, len(selected_models)), dtype=int) for _ in range(3)]
    for col_idx, model in enumerate(selected_models):
        for row_idx, layer in enumerate(LAYERS):
            user[row_idx, col_idx] = input_widgets[model][layer]['End User'].value
            sier[row_idx, col_idx] = input_widgets[model][layer]['SIer'].value
            csp[row_idx, col_idx] = input_widgets[model][layer]['CSP'].value
    return (sier, user, csp), None

# ==============================================================================
# 4. ipywidgets UIの構築とイベント処理
# ==============================================================================

# --- 4.1 UIウィジェットの作成 ---
title = widgets.HTML("<h2>クラウド責任共有モデル 可視化ツール</h2>")
description = widgets.HTML("<p>役割別の組織名や表示モデルを選択し、各タブで数値を入力後、各ボタンを押してください。</p>")
org_names_inputs = {
    'End User': widgets.Textarea(value='顧客企業名', description='End User:', style={'description_width': 'initial'}, layout=widgets.Layout(height='auto')),
    'SIer': widgets.Textarea(value='SIer企業名', description='SIer:', style={'description_width': 'initial'}, layout=widgets.Layout(height='auto')),
    'CSP': widgets.Textarea(value='クラウド事業者名', description='CSP:', style={'description_width': 'initial'}, layout=widgets.Layout(height='auto'))
}
org_box = widgets.VBox([widgets.Label(value="役割別の組織名:"), org_names_inputs['End User'], org_names_inputs['SIer'], org_names_inputs['CSP']], layout=widgets.Layout(margin='20px 0 0 0'))
model_selector = widgets.SelectMultiple(options=MODELS, value=MODELS, description='表示モデル:', style={'description_width': 'initial'}, layout=widgets.Layout(width='95%'))
button_layout = widgets.Layout(width='160px')
update_button = widgets.Button(description="グラフを更新", button_style='primary', layout=button_layout)
export_csv_button = widgets.Button(description="CSVダウンロード準備", button_style='success', layout=button_layout)
gsheet_name_input = widgets.Text(value='クラウド責任共有モデル', description='ファイル名の接頭辞:', style={'description_width': 'initial'})
export_gsheet_button = widgets.Button(description="Google Drive連携", button_style='warning', layout=button_layout)
status_display = widgets.HTML("")
download_link_display = widgets.HTML(value="")
input_widgets = {}
tab_children = []
for model in MODELS:
    layer_inputs = []; input_widgets[model] = {}
    for layer in LAYERS:
        layer_index = LAYERS.index(layer)
        s_val, u_val, c_val = INITIAL_VALUES[model][layer_index]
        u_input = widgets.IntText(value=u_val, description="End User %", style={'description_width': 'initial'})
        s_input = widgets.IntText(value=s_val, description="SIer %", style={'description_width': 'initial'})
        c_input = widgets.IntText(value=c_val, description="CSP %", style={'description_width': 'initial'})
        note_input = widgets.Textarea(value="", placeholder="補足事項があれば入力...", layout=widgets.Layout(height='60px'))
        input_widgets[model][layer] = {'End User': u_input, 'SIer': s_input, 'CSP': c_input, 'Note': note_input}
        layer_group = widgets.VBox([widgets.HTML(f"<b>{layer}</b>"), u_input, s_input, c_input, note_input])
        layer_inputs.append(layer_group)
    tab_children.append(widgets.VBox(layer_inputs))
input_tabs = widgets.Tab(); input_tabs.children = tab_children
for i, model in enumerate(MODELS): input_tabs.set_title(i, model)
chart_output = widgets.Output()

# --- 4.2 イベントハンドラ関数の定義 ---
def on_update_button_clicked(b):
    """「グラフを更新」ボタンが押された時の処理"""
    chart_output.clear_output(wait=True)
    download_link_display.value = ""
    selected_models = list(model_selector.value)
    selected_models.sort(key=MODELS.index)
    data, error = validate_and_get_data(selected_models, input_widgets)
    if error:
        status_display.value = f"<p style='color:red;'>{error}</p>"
        return
    org_names = {role: widget.value for role, widget in org_names_inputs.items()}
    sier_to_draw, user_to_draw, csp_to_draw = data
    with chart_output:
        fig = create_chart_figure(sier_to_draw, user_to_draw, csp_to_draw, selected_models, org_names)
        plt.show(fig)
        plt.close(fig)
    status_display.value = f"<p style='color:green;'>✅ グラフを正常に更新しました。（{', '.join(selected_models)}）</p>"

def on_export_csv_button_clicked(b):
    download_link_display.value = ""
    selected_models = list(model_selector.value)
    if not selected_models:
        status_display.value = "<p style='color:red;'>❌ エラー: 表示するサービスモデルを1つ以上選択してください。</p>"
        return
    
    # ファイル名接頭辞とタイムスタンプを取得
    base_name = gsheet_name_input.value
    if not base_name:
        status_display.value = f"<p style='color:red;'>❌ エラー: ファイル名の接頭辞を入力してください。</p>"
        return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{base_name}_{timestamp}.csv"
    
    # CSVデータを生成
    df = get_dataframe_from_inputs(input_widgets, selected_models)
    csv_str = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv_str.encode()).decode()
    
    # ダウンロードリンクを作成
    href = f'<a href="data:text/csv;base64,{b64}" download="{file_name}">「{file_name}」をダウンロード</a>'
    download_link_display.value = href
    status_display.value = "<p style='color:blue;'>📄 下記リンクからCSVをダウンロードしてください。</p>"

def on_export_to_gsheet_clicked(b):
    """「Google ドライブ連携」ボタンが押された時の処理"""
    status_display.value = "<p style='color:orange;'>⏳ Googleに認証し、Google Driveに書き込んでいます...</p>"
    download_link_display.value = ""
    
    selected_models = list(model_selector.value)
    selected_models.sort(key=MODELS.index)
    data, error = validate_and_get_data(selected_models, input_widgets)
    if error:
        status_display.value = f"<p style='color:red;'>{error}</p>"
        return
        
    try:
        # Google認証
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        drive_service = build('drive', 'v3', credentials=creds)
        
        # データ準備
        df_to_export = get_dataframe_from_inputs(input_widgets, selected_models)
        org_names = {role: widget.value for role, widget in org_names_inputs.items()}
        sier_data, user_data, csp_data = data
        fig_for_export = create_chart_figure(sier_data, user_data, csp_data, selected_models, org_names)
        
        # フォルダとファイルの名前をタイムスタンプ付きで生成
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = gsheet_name_input.value
        if not base_name:
            status_display.value = f"<p style='color:red;'>❌ エラー: ファイル名の接頭辞を入力してください。</p>"
            return
            
        folder_name = f"{base_name}_{timestamp}"
        sheet_name_with_ts = f"{base_name}_data_{timestamp}"
        image_name_with_ts = f"{base_name}_chart_{timestamp}.png"

        # Google Driveにフォルダを作成
        folder_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
        folder = drive_service.files().create(body=folder_metadata, fields='id, webViewLink').execute()
        folder_id = folder.get('id')
        folder_link = folder.get('webViewLink')

        # スプレッドシートを作成し、Google Driveのフォルダに移動
        sh = gc.create(sheet_name_with_ts)
        drive_service.files().update(fileId=sh.id, addParents=folder_id, removeParents='root').execute()
        sh.share(None, perm_type='anyone', role='reader')

        # Sheet1にデータを書き込み
        worksheet1 = sh.get_worksheet(0) or sh.add_worksheet(title="データ", rows="100", cols="20")
        worksheet1.clear()
        set_with_dataframe(worksheet1, df_to_export)
        
        # グラフ画像をPNGとしてメモリに保存し、Google Driveのフォルダにアップロード
        buf = io.BytesIO()
        fig_for_export.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig_for_export)
        buf.seek(0)
        
        file_metadata = {'name': image_name_with_ts, 'mimeType': 'image/png', 'parents': [folder_id]}
        media = MediaIoBaseUpload(buf, mimetype='image/png')
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id, webContentLink').execute()
        drive_service.permissions().create(fileId=file.get('id'), body={'type': 'anyone', 'role': 'reader'}).execute()
        
        status_display.value = f"<p style='color:green;'>✅ <a href='{folder_link}' target='_blank'>フォルダ「{folder_name}」への出力が完了しました。</a></p>"

    except Exception as e:
        status_display.value = f"<p style='color:red;'>❌ エラーが発生しました: {e}</p>"

# --- 4.3 イベントの接続 ---
update_button.on_click(on_update_button_clicked)
export_csv_button.on_click(on_export_csv_button_clicked)
export_gsheet_button.on_click(on_export_to_gsheet_clicked)

# --- 4.4 UI全体のレイアウト ---
control_panel = widgets.VBox([
    title,
    description,
    model_selector,
    widgets.HBox([update_button, export_csv_button, export_gsheet_button]),
    gsheet_name_input,
    status_display,
    download_link_display,
    org_box,
    widgets.HTML("<h3>調整パネル</h3>"),
    input_tabs
])
chart_panel = widgets.VBox([
    widgets.HTML("<h3>責任分担図</h3>"),
    chart_output,
])
app_layout = widgets.HBox([control_panel, chart_panel])

# ==============================================================================
# 5. アプリケーションの表示と初期化
# ==============================================================================
display(app_layout)
on_update_button_clicked(None)
try:
    from google.colab import output
    output.eval_js("new Promise(resolve => setTimeout(() => {document.querySelector('#output-area').scrollIntoView({ behavior: 'smooth', block: 'start' }); resolve();}, 200))")
