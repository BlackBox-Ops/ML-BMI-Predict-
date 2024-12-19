document.getElementById('bmiForm').addEventListener('submit', function(event) {
    event.preventDefault();

    // Ambil data dari form
    const gender = document.getElementById('gender').value;
    const height = document.getElementById('height').value;
    const weight = document.getElementById('weight').value;

    // Buat payload untuk request
    const payload = {
        gender: parseInt(gender),
        height: parseFloat(height),
        weight: parseFloat(weight)
    };

    // Kirim request ke endpoint Flask
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
        // Tampilkan hasil prediksi
        document.getElementById('result').innerText = `Prediction: ${data.prediction}`;
    })
    .catch(error => {
        document.getElementById('result').innerText = `Error: ${error}`;
    });
});
