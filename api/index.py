from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/similar-players', methods=['POST'])
def find_similar_players_route():
    data = request.get_json()
    # Call your find_similar_players logic here
    return jsonify({"message": "Success!", "received": data})
