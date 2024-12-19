document.getElementById('bmiForm').addEventListener('submit', function (event) {
    event.preventDefault();

    // Ambil nilai dari form
    const gender = document.getElementById('gender').value;
    const height = document.getElementById('height').value;
    const weight = document.getElementById('weight').value;

    // Periksa apakah semua input terisi
    if (!gender || !height || !weight) {
        showAlert('danger', 'Please fill out all fields.');
        return;
    }

    // Tampilkan indikator loading
    showAlert('info', '<i class="fas fa-spinner fa-spin"></i> Predicting...');

    // Payload untuk API
    const payload = {
        gender: parseInt(gender),
        height: parseFloat(height),
        weight: parseFloat(weight),
    };

    // Kirim request ke Flask API
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.error) {
                showAlert('danger', `Error: ${data.error}`);
                document.getElementById('result').value = '';
            } else {
                showAlert('success', 'Prediction successful!');
                document.getElementById('result').value = data.prediction;
            }
        })
        .catch((error) => {
            showAlert('danger', `Error: ${error}`);
            document.getElementById('result').value = '';
        });
});

// Fungsi untuk menampilkan alert
function showAlert(type, message) {
    const alertDiv = document.getElementById('alert');
    alertDiv.className = `alert alert-${type} d-block`;
    alertDiv.innerHTML = message;

    // Sembunyikan alert setelah 5 detik
    setTimeout(() => {
        alertDiv.className = 'alert d-none';
    }, 5000);
}
