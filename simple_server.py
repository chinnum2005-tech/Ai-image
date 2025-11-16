from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/test')
def test():
    return jsonify({"status": "success", "message": "Simple server is working!"})

if __name__ == '__main__':
    print("=== Starting Simple Test Server ===")
    print("Try accessing: http://localhost:5007/test")
    print("Press Ctrl+C to stop the server")
    app.run(host='0.0.0.0', port=5007, debug=True)
