import os
import csv
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from PIL import Image


app = Flask(__name__)
app.secret_key = "some_secret_key"  # needed for flash messages

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")


KAGGLE_STYLES_FILE = os.path.join(DATA_DIR, "styles.csv")

WARDROBE_FILE = os.path.join(DATA_DIR, "wardrobe_items.csv")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_log.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------- ENCODING MAPPINGS ----------------
# Used for KNN feature vector

SKIN_TONE_MAP = {
    "fair": 0,
    "medium": 1,
    "tan": 2,
    "deep": 3,
}

BODY_SHAPE_MAP = {
    "rectangle": 0,
    "pear": 1,
    "hourglass": 2,
    "inverted_triangle": 3,
}

WEATHER_MAP = {
    "cold": 0,
    "normal": 1,
    "hot": 2,
    "rainy": 3,
}

OCCASION_MAP = {
    "casual": 0,
    "office": 1,
    "party": 2,
    "date": 3,
    "festival": 4,
}

STYLE_MAP = {
    "minimal": 0,
    "sporty": 1,
    "street": 2,
    "classic": 3,
    "party": 4,
}

COMFORT_PRIORITY_MAP = {
    "comfort_first": 0,
    "style_first": 1,
}

FIT_MAP = {
    "loose": 0,
    "regular": 1,
    "slim": 2,
}

BUDGET_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

GENDER_MAP = {
    "male": 0,
    "female": 1,
    "other": 2,
}


def init_wardrobe_file():
    if os.path.exists(WARDROBE_FILE):
        return
    with open(WARDROBE_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "item_id", "item_name", "category", "color", "fit", "price"
        ])


def init_feedback_file():
    if os.path.exists(FEEDBACK_FILE):
        return
    with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "outfit_id", "like_dislike",
            "skin_tone", "body_shape", "weather", "occasion",
            "style_personality", "comfort_priority", "fit",
            "budget_level", "gender"
        ])


init_wardrobe_file()
init_feedback_file()


def encode_skin_tone(name):
    return SKIN_TONE_MAP.get(name, 1)


def encode_body_shape(name):
    return BODY_SHAPE_MAP.get(name, 0)


def encode_weather(label):
    return WEATHER_MAP.get(label, 1)


def encode_occasion(label):
    return OCCASION_MAP.get(label, 0)


def encode_style(label):
    return STYLE_MAP.get(label, 0)


def encode_comfort_priority(label):
    return COMFORT_PRIORITY_MAP.get(label, 0)


def encode_fit(label):
    return FIT_MAP.get(label, 1)


def encode_budget(label):
    return BUDGET_MAP.get(label, 1)


def encode_gender(label):
    return GENDER_MAP.get(label, 0)


# --------------- FEATURE 4: WEATHER CLASSIFICATION ----------------

def classify_weather_from_temp(temp_c, is_rainy):
    """Return weather label string."""
    if is_rainy:
        return "rainy"
    if temp_c is None:
        return "normal"
    if temp_c < 18:
        return "cold"
    elif 18 <= temp_c <= 28:
        return "normal"
    else:
        return "hot"


def classify_body_shape_from_measurements(shoulder, bust, waist, hip):
    """
    Very simple rule-based body shape classification.
    shoulder, bust, waist, hip in cm.
    """
    if any(v is None for v in [shoulder, bust, waist, hip]):
        return "rectangle"

    shoulder_hip_diff = shoulder - hip
    waist_hip_ratio = waist / hip if hip != 0 else 1

    if abs(shoulder_hip_diff) < 3 and waist_hip_ratio > 0.85:
        return "rectangle"
    if hip - shoulder > 4:
        return "pear"
    if abs(shoulder_hip_diff) <= 3 and waist_hip_ratio < 0.8:
        return "hourglass"
    if shoulder - hip > 4:
        return "inverted_triangle"

    return "rectangle"



def detect_skin_tone_from_image(image_path):
    """
    Simple prototype: take center crop of the image,
    average RGB, then map brightness to a skin tone category.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        crop_size_w = int(w * 0.4)
        crop_size_h = int(h * 0.4)
        left = (w - crop_size_w) // 2
        top = (h - crop_size_h) // 2
        right = left + crop_size_w
        bottom = top + crop_size_h
        crop = img.crop((left, top, right, bottom))
        arr = np.array(crop)
        avg = arr.mean(axis=(0, 1))  # [R, G, B]
        brightness = avg.mean()

        if brightness > 200:
            return "fair"
        elif brightness > 150:
            return "medium"
        elif brightness > 100:
            return "tan"
        else:
            return "deep"
    except Exception:
        return "medium"



def normalize_gender_kaggle(g):
    if not isinstance(g, str):
        return "other"
    g = g.strip().lower()
    if g in ["men", "boys", "male"]:
        return "male"
    if g in ["women", "girls", "female"]:
        return "female"
    return "other"


def map_season_to_weather(season):
    if not isinstance(season, str):
        return "normal"
    s = season.strip().lower()
    if s == "summer":
        return "hot"
    if s == "winter":
        return "cold"
    if s in ["spring", "fall", "autumn"]:
        return "normal"
    return "normal"


def map_usage_to_occasion(usage):
    if not isinstance(usage, str):
        return "casual"
    u = usage.strip().lower()
    if "casual" in u:
        return "casual"
    if "formal" in u or "work" in u:
        return "office"
    if "party" in u or "evening" in u:
        return "party"
    if "ethnic" in u or "festive" in u:
        return "festival"
    if "sports" in u or "active" in u:
        return "casual"
    return "casual"


def map_article_to_style(article_type, sub_category):
    text = ""
    if isinstance(article_type, str):
        text += article_type.lower() + " "
    if isinstance(sub_category, str):
        text += sub_category.lower()

    if any(k in text for k in ["track", "sports", "jogger", "training", "tights"]):
        return "sporty"
    if any(k in text for k in ["hoodie", "sweatshirt", "jeans", "tshirt", "t-shirt", "jacket"]):
        return "street"
    if any(k in text for k in ["kurta", "saree", "sari", "ethnic", "sherwani", "lehenga"]):
        return "classic"
    if any(k in text for k in ["dress", "gown", "skirt", "heels", "pumps"]):
        return "party"
    return "minimal"


def infer_skin_tone_from_base_colour(base_colour):
    if not isinstance(base_colour, str):
        return "medium"
    c = base_colour.strip().lower()
    warm = ["red", "maroon", "orange", "yellow", "gold", "brown", "rust", "magenta", "khaki"]
    cool = ["blue", "navy", "green", "sea green", "teal", "turquoise", "purple"]
    if any(w in c for w in warm):
        return "tan"
    if any(w in c for w in cool):
        return "fair"
    return "medium"


def infer_body_shape_from_article(article_type, sub_category):
    text = ""
    if isinstance(article_type, str):
        text += article_type.lower() + " "
    if isinstance(sub_category, str):
        text += sub_category.lower()

    if any(k in text for k in ["bodycon", "slim", "skinny", "fit & flare"]):
        return "hourglass"
    if any(k in text for k in ["a-line", "flared", "bootcut"]):
        return "pear"
    if any(k in text for k in ["straight", "shift"]):
        return "rectangle"
    return "rectangle"



def build_feature_vector(skin_tone, body_shape, weather, occasion,
                         style_personality, comfort_priority, fit,
                         budget_level, gender):
    """Return numeric list used in KNN."""
    return [
        encode_skin_tone(skin_tone),
        encode_body_shape(body_shape),
        encode_weather(weather),
        encode_occasion(occasion),
        encode_style(style_personality),
        encode_comfort_priority(comfort_priority),
        encode_fit(fit),
        encode_budget(budget_level),
        encode_gender(gender),
    ]



def get_accessories_suggestions(body_shape, occasion, style_personality):
    suggestions = []

    if body_shape == "pear":
        suggestions.append("Statement necklaces or earrings to draw attention upwards.")
        suggestions.append("Structured jackets to balance hips.")
    elif body_shape == "hourglass":
        suggestions.append("Belts to highlight your waist.")
    elif body_shape == "inverted_triangle":
        suggestions.append("Chunky bracelets and rings to balance broader shoulders.")
    else:
        suggestions.append("Layered necklaces and scarves to add curves visually.")

    if occasion in ["party", "festival"]:
        suggestions.append("Add some shimmer jewellery or a bold statement piece.")
        suggestions.append("Carry a clutch or small sling bag.")
    elif occasion == "office":
        suggestions.append("Minimal jewellery like studs, watch, and a structured bag.")
    else:
        suggestions.append("Simple studs or hoops and a daily-wear watch.")

    if style_personality == "sporty":
        suggestions.append("Smartwatch / fitness band and sporty cap.")
    elif style_personality == "street":
        suggestions.append("Chunky sneakers, caps, layered chains.")
    elif style_personality == "minimal":
        suggestions.append("Delicate chain and simple studs.")
    elif style_personality == "party":
        suggestions.append("Bold earrings and high heels for extra glam.")

    return suggestions


def get_color_palette_recommendations(skin_tone):
    if skin_tone == "fair":
        best = ["Soft pastels", "Cool blues", "Rose pink", "Mint"]
        avoid = ["Very neon shades that overpower", "Extremely pale yellows"]
    elif skin_tone == "medium":
        best = ["Earth tones (olive, camel)", "Teal", "Coral", "Mustard"]
        avoid = ["Neon green", "Overly cool greyish tones"]
    elif skin_tone == "tan":
        best = ["Burnt orange", "Mustard", "Bottle green", "Maroon"]
        avoid = ["Ashy greys", "Very light beige close to skin"]
    else:
        best = ["Jewel tones (emerald, royal blue)", "Gold", "Fuchsia", "Deep reds"]
        avoid = ["Very light pastels that may look chalky"]

    return best, avoid



def get_makeup_suggestions(skin_tone, occasion):
    base = []
    lips = []
    eyes = []

    if skin_tone in ["fair", "medium"]:
        base.append("Use a foundation with neutral or slightly warm undertone.")
    else:
        base.append("Use a foundation with warm / golden undertone to avoid grey cast.")

    if occasion == "office":
        lips.append("Nude, soft pink, or dusty rose lip shades.")
        eyes.append("Light brown eyeshadow, thin eyeliner, and mascara.")
    elif occasion in ["party", "festival"]:
        lips.append("Bold lip shades like red, berry, or deep nude.")
        eyes.append("Smokey eyes or shimmer on lids with kajal and mascara.")
    else:
        lips.append("Soft pink, peach, or MLBB shades.")
        eyes.append("Light shimmer or matte browns with mascara.")

    return {
        "base": base,
        "lips": lips,
        "eyes": eyes,
    }



def load_wardrobe_items():
    items = []
    if not os.path.exists(WARDROBE_FILE):
        return items
    with open(WARDROBE_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    return items


def add_wardrobe_item(item_name, category, color, fit, price):
    existing = load_wardrobe_items()
    next_id = 1
    if existing:
        next_id = max(int(x["item_id"]) for x in existing) + 1

    with open(WARDROBE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([next_id, item_name, category, color, fit, price])


def pick_wardrobe_combination(wardrobe_items):
    tops = [i for i in wardrobe_items if i["category"].lower() in
            ["top", "shirt", "tshirt", "kurta", "blouse", "dress"]]
    bottoms = [i for i in wardrobe_items if i["category"].lower() in
               ["jeans", "trousers", "bottom", "skirt", "leggings"]]
    shoes = [i for i in wardrobe_items if i["category"].lower() in
             ["shoes", "sneakers", "heels", "sandals", "footwear"]]

    outfit = {}
    if tops:
        outfit["top"] = tops[0]
    if bottoms:
        outfit["bottom"] = bottoms[0]
    if shoes:
        outfit["footwear"] = shoes[0]

    return outfit if outfit else None


def append_feedback(outfit_id, like_dislike, user_features_dict):
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            outfit_id,
            like_dislike,
            user_features_dict.get("skin_tone"),
            user_features_dict.get("body_shape"),
            user_features_dict.get("weather"),
            user_features_dict.get("occasion"),
            user_features_dict.get("style_personality"),
            user_features_dict.get("comfort_priority"),
            user_features_dict.get("fit"),
            user_features_dict.get("budget_level"),
            user_features_dict.get("gender"),
        ])



def build_explanation(user_data, outfit_row, weather_label, color_best):
    explanation = []
    explanation.append(f"This outfit matches your **{user_data['style_personality']}** style preference.")
    explanation.append(f"It is suitable for a **{user_data['body_shape']}** body shape.")
    explanation.append(f"The outfit works well for **{user_data['occasion']}** occasions.")
    explanation.append(f"Recommended for **{weather_label}** weather conditions.")
    explanation.append(
        f"The colors suggested go well with your **{user_data['skin_tone']}** skin tone "
        f"(e.g. {', '.join(color_best[:3])})."
    )
    explanation.append(f"It respects your budget preference (**{user_data['budget_level']}**).")
    return " ".join(explanation)



def load_outfits_df():
    """
    Read Kaggle styles.csv and convert into our internal outfit structure.
    HUGE dataset, no manual CSV creation.
    """
    if not os.path.exists(KAGGLE_STYLES_FILE):
        raise FileNotFoundError(
            f"styles.csv not found in {DATA_DIR}. "
            f"Download it from Kaggle (Fashion Product Images Dataset) and place it there."
        )

    raw = pd.read_csv(KAGGLE_STYLES_FILE)

    required_cols = [
        "id", "gender", "masterCategory", "subCategory",
        "articleType", "baseColour", "season", "usage", "productDisplayName"
    ]
    for col in required_cols:
        if col not in raw.columns:
            raise ValueError(f"Expected column '{col}' not found in Kaggle styles.csv")

    processed_rows = []

    for _, r in raw.iterrows():
        gender_norm = normalize_gender_kaggle(r["gender"])
        if gender_norm == "other":
            # skip unisex/unknown for simplicity
            continue

        product_name = str(r["productDisplayName"])
        base_colour = r["baseColour"]
        season = r["season"]
        usage = r["usage"]
        master_cat = r["masterCategory"]
        sub_cat = r["subCategory"]
        article_type = r["articleType"]

        weather = map_season_to_weather(season)
        occasion = map_usage_to_occasion(usage)
        style_personality = map_article_to_style(article_type, sub_cat)
        skin_tone_best = infer_skin_tone_from_base_colour(base_colour)
        body_shape_best = infer_body_shape_from_article(article_type, sub_cat)

        budget_level = "medium"  # all medium; user filter still works vs low/high
        comfort_priority = "comfort_first"
        fit = "regular"

        notes = f"{master_cat} / {sub_cat} / {article_type} ({base_colour})"

        outfit = {
            "outfit_id": int(r["id"]),
            "name": product_name,
            "gender": gender_norm,
            "skin_tone": skin_tone_best,
            "body_shape": body_shape_best,
            "weather": weather,
            "occasion": occasion,
            "style_personality": style_personality,
            "comfort_priority": comfort_priority,
            "fit": fit,
            "budget_level": budget_level,
            "top": product_name,
            "bottom": "-",
            "footwear": "",
            "notes": notes,
        }
        processed_rows.append(outfit)

    df = pd.DataFrame(processed_rows)
    return df


@app.route("/", methods=["GET"])
def index():
    wardrobe_items = load_wardrobe_items()
    return render_template("index.html", wardrobe_items=wardrobe_items)


@app.route("/add_wardrobe_item", methods=["POST"])
def add_wardrobe_item_route():
    item_name = request.form.get("item_name", "").strip()
    category = request.form.get("category", "").strip()
    color = request.form.get("color", "").strip()
    fit = request.form.get("fit", "regular").strip()
    price = request.form.get("price", "0").strip()

    if item_name and category:
        add_wardrobe_item(item_name, category, color, fit, price)
        flash("Wardrobe item added successfully!", "success")
    else:
        flash("Item name and category are required.", "danger")

    return redirect(url_for("index"))


@app.route("/recommend", methods=["POST"])
def recommend():

    gender = request.form.get("gender", "male")
    style_personality = request.form.get("style_personality", "minimal")
    comfort_priority = request.form.get("comfort_priority", "comfort_first")
    fit = request.form.get("fit", "regular")
    occasion = request.form.get("occasion", "casual")
    budget_level = request.form.get("budget_level", "medium")

    temp_str = request.form.get("temperature", "").strip()
    rainy_flag = request.form.get("is_rainy", "no")
    temp_c = None
    if temp_str:
        try:
            temp_c = float(temp_str)
        except ValueError:
            temp_c = None
    is_rainy = (rainy_flag == "yes")
    weather_label = classify_weather_from_temp(temp_c, is_rainy)

    skin_tone_manual = request.form.get("skin_tone_manual", "medium")
    face_file = request.files.get("face_image")

    if face_file and face_file.filename != "":
        face_path = os.path.join(UPLOAD_DIR, "face_" + face_file.filename)
        face_file.save(face_path)
        detected_skin_tone = detect_skin_tone_from_image(face_path)
        skin_tone = detected_skin_tone
    else:
        skin_tone = skin_tone_manual

  
    shoulder = request.form.get("shoulder", "").strip()
    bust = request.form.get("bust", "").strip()
    waist = request.form.get("waist", "").strip()
    hip = request.form.get("hip", "").strip()

    def to_float(val):
        try:
            return float(val)
        except ValueError:
            return None

    shoulder_val = to_float(shoulder)
    bust_val = to_float(bust)
    waist_val = to_float(waist)
    hip_val = to_float(hip)

    body_shape_manual = request.form.get("body_shape_manual", "rectangle")

    if all(v is not None for v in [shoulder_val, bust_val, waist_val, hip_val]):
        body_shape = classify_body_shape_from_measurements(shoulder_val, bust_val, waist_val, hip_val)
    else:
        body_shape = body_shape_manual
-
    user_vector = build_feature_vector(
        skin_tone=skin_tone,
        body_shape=body_shape,
        weather=weather_label,
        occasion=occasion,
        style_personality=style_personality,
        comfort_priority=comfort_priority,
        fit=fit,
        budget_level=budget_level,
        gender=gender,
    )

    user_data = {
        "skin_tone": skin_tone,
        "body_shape": body_shape,
        "weather": weather_label,
        "occasion": occasion,
        "style_personality": style_personality,
        "comfort_priority": comfort_priority,
        "fit": fit,
        "budget_level": budget_level,
        "gender": gender,
    }

    df = load_outfits_df()

    outfit_vectors = []
    for _, row in df.iterrows():
        vec = build_feature_vector(
            skin_tone=row["skin_tone"],
            body_shape=row["body_shape"],
            weather=row["weather"],
            occasion=row["occasion"],
            style_personality=row["style_personality"],
            comfort_priority=row["comfort_priority"],
            fit=row["fit"],
            budget_level=row["budget_level"],
            gender=row["gender"],
        )
        outfit_vectors.append(vec)

    X = np.array(outfit_vectors)
    neigh = NearestNeighbors(n_neighbors=min(10, len(X)), metric="euclidean")
    neigh.fit(X)

    distances, indices = neigh.kneighbors([user_vector])

  
    for idx in indices[0]:
        row = df.iloc[idx].to_dict()

        if budget_level == "low" and row["budget_level"] == "high":
            continue
        if budget_level == "high" and row["budget_level"] == "low":
            continue

        if row["style_personality"] != style_personality:
            continue
        if row["occasion"] != occasion:
            continue
        if row["weather"] != weather_label and weather_label != "rainy":
            continue

        recommended_outfits.append(row)

    if not recommended_outfits:
        for idx in indices[0]:
            recommended_outfits.append(df.iloc[idx].to_dict())

    accessories_suggestions = get_accessories_suggestions(body_shape, occasion, style_personality)
    best_colors, avoid_colors = get_color_palette_recommendations(skin_tone)
    makeup_suggestions = None
    if gender == "female":
        makeup_suggestions = get_makeup_suggestions(skin_tone, occasion)

 
    wardrobe_items = load_wardrobe_items()
    use_wardrobe = request.form.get("use_wardrobe", "no") == "yes"
    wardrobe_outfit = None
    if use_wardrobe and wardrobe_items:
        wardrobe_outfit = pick_wardrobe_combination(wardrobe_items)

    explanation = build_explanation(user_data, recommended_outfits[0], weather_label, best_colors)

    return render_template(
        "result.html",
        user=user_data,
        outfits=recommended_outfits,
        accessories_suggestions=accessories_suggestions,
        best_colors=best_colors,
        avoid_colors=avoid_colors,
        makeup_suggestions=makeup_suggestions,
        wardrobe_outfit=wardrobe_outfit,
        explanation=explanation,
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    outfit_id = request.form.get("outfit_id")
    like_dislike = request.form.get("like_dislike")

    skin_tone = request.form.get("skin_tone")
    body_shape = request.form.get("body_shape")
    weather = request.form.get("weather")
    occasion = request.form.get("occasion")
    style_personality = request.form.get("style_personality")
    comfort_priority = request.form.get("comfort_priority")
    fit = request.form.get("fit")
    budget_level = request.form.get("budget_level")
    gender = request.form.get("gender")

    user_features_dict = {
        "skin_tone": skin_tone,
        "body_shape": body_shape,
        "weather": weather,
        "occasion": occasion,
        "style_personality": style_personality,
        "comfort_priority": comfort_priority,
        "fit": fit,
        "budget_level": budget_level,
        "gender": gender,
    }

    append_feedback(outfit_id, like_dislike, user_features_dict)
    flash("Feedback stored successfully (system is learning)!", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    # install deps: pip install flask scikit-learn pillow pandas
    app.run(debug=True)

