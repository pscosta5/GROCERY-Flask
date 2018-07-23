from flask import Flask
from flask import render_template
from flask import request
import recommendation_engine as rec

app = Flask(__name__)

@app.route("/index.html", methods=['POST', 'GET'])
def index():

    if request.method == "POST":
        checkbox_hotdog = request.form.get('hotdog')
        checkbox_protein = request.form.get('protein')
        checkbox_tofu = request.form.get('tofu')
        checkbox_carrot = request.form.get('carrot')
        checkbox_spice = request.form.get('spice')
        checkbox_baguettes = request.form.get('baguettes')
        checkbox_muffins = request.form.get('muffins')
        checkbox_toast = request.form.get('toast')
        checkbox_steak = request.form.get('steak')
        checkbox_raspberry = request.form.get('raspberry')
        checkbox_strawberry = request.form.get('strawberry')
        checkbox_donut = request.form.get('donut')
        checkbox_pancakes = request.form.get('pancakes')
        checkbox_salad = request.form.get('salad')
        checkbox_orange = request.form.get('orange')
        checkbox_onion = request.form.get('onion')
        checkbox_churro = request.form.get('churro')
        checkbox_pepper = request.form.get('pepper')
        checkbox_salmon = request.form.get('salmon')
        checkbox_blueberry = request.form.get('blueberry')
        checkbox_roast = request.form.get('roast-beef')

        checkbox_list = [checkbox_hotdog, checkbox_protein, checkbox_tofu, checkbox_carrot, checkbox_spice, checkbox_baguettes, \
        checkbox_muffins, checkbox_toast, checkbox_steak, checkbox_raspberry, \
        checkbox_strawberry, checkbox_donut, checkbox_pancakes, checkbox_salad, checkbox_orange, \
        checkbox_onion, checkbox_churro, checkbox_pepper, checkbox_salmon, checkbox_blueberry, checkbox_roast]
        selected_items = ""
        for check in checkbox_list:
            if check:
                selected_items += " " + check.strip() + " " + check.strip() + " " + check.strip()
        if selected_items:
            selected_items = selected_items.strip()
            recommended_products = rec.product_id_to_name(rec.get_recommended_products(selected_items, 200, include_prev=False, debug=False))
            chosen_products = rec.product_id_to_name([int(i) for i in selected_items.split()])
            item1 = recommended_products[0]
            item2 = recommended_products[1]
            item3 = recommended_products[2]
            item4 = recommended_products[3]
            item5 = recommended_products[4]
            item6 = recommended_products[5]
            item7 = recommended_products[6]
            item8 = recommended_products[7]
            item9 = recommended_products[8]
            item10 = recommended_products[9]
        else:
            recommended_products = None
            chosen_products = None
            item1 = None
            item2 = None
            item3 = None
            item4 = None
            item5 = None
            item6 = None
            item7 = None
            item8 = None
            item9 = None
            item10 = None
        return render_template("recommendations.html", passthrough=recommended_products, \
        selection=chosen_products, item1=item1, item2=item2, item3=item3, item4=item4, \
        item5=item5, item6=item6, item7=item7, item8=item8, item9=item9, item10=item10)
    else:
        return render_template("index.html")

@app.route("/insights.html", methods=['POST', 'GET'])
def insights():
    return render_template("insights.html")

@app.route("/about.html", methods=['POST', 'GET'])
def about():
    return render_template("about.html")

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    return render_template("recommendations.html")

@app.route("/test_rec", methods=['POST', 'GET'])
def test_rec():
    return render_template("test_rec.html")


# Change this before submitting
if __name__ == "__main__":
    app.run(debug=False)
