import json
import time
from typing import Literal
from flask import Flask, request, jsonify, send_file
from google import genai
from google.genai import types
from pydantic import BaseModel

client = genai.Client()
app = Flask(__name__)

# AIの出力選択肢をシステムレベルで厳格に固定
CompositionStyle = Literal[
    "low_angle", "high_angle", "close_up", "wide_scenery", "diagonal_action", "frame_in_frame"
]
SubjectPosition = Literal[
    "left", "center", "right", "bottom_left", "bottom_right"
]
SubjectScale = Literal[
    "face_only", "bust_up", "waist_up", "knee_up", "full_body"
]
GazeDirection = Literal[
    "left_to_right", "right_to_left", "bottom_to_top", "diagonal_up", "converge_center"
]
VisualElement = Literal[
    "sun_or_moon", "magic_or_light", "speed_lines", "architecture_or_fence", "nature_or_clouds", "foreground_blur"
]

class CompositionPlan(BaseModel):
    tag: str
    title: str
    angle: str
    layout: str
    gaze: str
    composition_style: CompositionStyle
    subject_position: SubjectPosition
    subject_scale: SubjectScale
    gaze_direction: GazeDirection
    visual_element: VisualElement

class CompositionResponse(BaseModel):
    plans: list[CompositionPlan]

SCALE_DESC = {
    "face_only": "顔のみ（表情・瞳の描き込み重視）",
    "bust_up": "バストアップ（胸より上）",
    "waist_up": "腰より上（ウエストショット）",
    "knee_up": "膝より上（ニーショット）",
    "full_body": "全身（足先まで画面に収める立ち絵）",
}

MOOD_DESC = {
    "cute": "かわいい・愛らしい雰囲気",
    "cool": "クール・スタイリッシュな雰囲気",
    "emotional": "エモい・切なさや空気感のある雰囲気",
    "dynamic": "迫力・アクション・臨場感のある雰囲気",
}

@app.route("/")
def home():
    return send_file("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    body = request.json or {}
    prompt = body.get("prompt", "").strip()
    scale = body.get("scale", "other")
    mood = body.get("mood", "other")

    if not prompt and scale == "other" and mood == "other":
        return jsonify({"error": "描画範囲・雰囲気を選ぶか、自由記述欄に条件を入力してください。"}), 400

    # UIで選択された条件を構造化テキストに組み立て
    conditions = []
    if scale in SCALE_DESC:
        conditions.append(f"・指定された描画範囲: {SCALE_DESC[scale]}（※3案すべて subject_scale='{scale}' に固定すること）")
    else:
        conditions.append("・描画範囲: 自由記述の内容から判断（指定がなければ3案で異なる範囲を提案）")

    if mood in MOOD_DESC:
        conditions.append(f"・指定された雰囲気・テイスト: {MOOD_DESC[mood]}")
    else:
        conditions.append("・雰囲気・テイスト: 自由記述の内容に合わせる")

    if prompt:
        conditions.append(f"・自由記述の詳細条件: {prompt}")
    else:
        conditions.append("・自由記述の詳細条件: （未入力のため、指定された描画範囲と雰囲気に合う魅力的なシチュエーションを自動補完してください）")

    structured_request = "\n".join(conditions)
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=f"""
以下の条件をもとに、イラスト制作の構図プランを3つ提案してください。

【ユーザーの指定条件】
{structured_request}

【重要ルール】
1. 描画範囲が指定されている場合は、3案すべてその subject_scale に厳密に従うこと。
2. 自由記述が短い・抽象的な場合でもエラーや単調な提案にせず、AI側で以下の3つの切り口にシチュエーションを自動補完してバリエーションを出すこと：
   - 提案01: 王道・キャラクターの魅力がストレートに伝わる構図
   - 提案02: 光・背景・小物などの演出で空気感を高める構図
   - 提案03: カメラアングルや動き（傾き・パース）で印象づける構図
3. 3つの提案は必ず異なる composition_style・subject_position・gaze_direction・visual_element を組み合わせること。
""",
                config=types.GenerateContentConfig(
                    system_instruction="あなたはプロのイラスト構図アドバイザーです。ユーザーの指定条件と自由記述を尊重し、具体的なシチュエーションを補完しながら3つの異なる構図バリエーションを提案してください。",
                    response_mime_type="application/json",
                    response_schema=CompositionResponse,
                )
            )

            # 安全フィルタ等で空レスポンスになった場合の検知
            if not response or not response.text:
                return jsonify({
                    "error": "条件が抽象的すぎるため、安全フィルタ等により生成がスキップされました。自由記述欄に「衣装・場所・時間帯」などを少し書き足してお試しください。"
                }), 400

            data = json.loads(response.text)
            return jsonify(data)

        except Exception as e:
            err_msg = str(e)
            is_busy = any(k in err_msg for k in ["503", "UNAVAILABLE", "high demand", "overloaded"])
            is_rate_limit = any(k in err_msg for k in ["429", "RESOURCE_EXHAUSTED", "quota"])
            is_auth_error = any(k in err_msg for k in ["API_KEY_INVALID", "API key not valid", "401", "403"])

            if is_busy and attempt < max_retries:
                time.sleep(2)
                continue

            if is_busy:
                return jsonify({
                    "error": "現在AIサーバーにアクセスが集中しており混み合っています。30秒ほど時間を置いてから再度お試しください（右上の「プレビュー」でも動作を体験できます）。"
                }), 503
            if is_rate_limit:
                return jsonify({
                    "error": "短時間の利用回数上限に達しました。1分ほど待ってから再度お試しください。"
                }), 429
            if is_auth_error:
                return jsonify({
                    "error": "APIキーの認証に失敗しました。サーバー側のAPIキー設定を確認してください。"
                }), 401

            return jsonify({
                "error": "構図の生成中に一時的なエラーが発生しました。自由記述にシーンの補足を入れるか、少し時間を置いて再度お試しください。"
            }), 500

if __name__ == "__main__":
    app.run(debug=True)