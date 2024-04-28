from flask import Flask, request, jsonify
from cartoonizer import cartoon

app = Flask(__name__)

@app.route('/cartoonize', methods=['POST'])
def post_string():
    if request.method == 'POST':
        # Get the string from the request
        data = request.json
        input_string = data.get('base64string')

        outputBase64 = cartoon(input_string)

        # Return the string in the response
        return jsonify({'output_string': outputBase64})
    else:
        return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)