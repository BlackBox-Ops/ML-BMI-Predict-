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

    // Tampilkan indikator loading dan animasi pada icon dan title
    showAlert('info', '<i class="fas fa-spinner fa-spin"></i> Predicting...');
    const wizardIcon = document.getElementById('wizardIcon');
    const title = document.getElementById('title');

    // Tambahkan animasi icon spin
    wizardIcon.classList.add('fa-spin');

    // Ubah tulisan menjadi "Analyzer"
    changeTitleText("Analyzer");

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
        })
        .finally(() => {
            // Kembalikan title ke "BMI Wizard" dan reset hasil setelah 5 detik
            setTimeout(() => {
                wizardIcon.classList.remove('fa-spin');
                changeTitleText("BMI Wizard");
                resetResult();
            }, 5000);
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
    }, 3000);
}

// Fungsi untuk mengubah teks title huruf per huruf
function changeTitleText(newText) {
    const title = document.getElementById('title');
    title.innerHTML = ''; // Kosongkan teks lama

    // Tambahkan huruf per huruf dengan animasi
    [...newText].forEach((char, index) => {
        const span = document.createElement('span');
        span.textContent = char;
        span.style.animationDelay = `${index * 0.1}s`;
        title.appendChild(span);
    });
}

// Fetch and update the prediction table
function updateTable() {
    fetch('/history')
        .then((response) => response.json())
        .then((data) => {
            const tableBody = document.getElementById('predictionTable');
            tableBody.innerHTML = '';
            data.forEach((item, index) => {
                const row = `<tr>
                    <td>${index + 1}</td>
                    <td>${item.gender}</td>
                    <td>${item.height}</td>
                    <td>${item.weight}</td>
                    <td>${item.result}</td>
                </tr>`;
                tableBody.innerHTML += row;
            });
        });
}

function showAlert(type, message) {
    const alertDiv = document.getElementById('alert');
    alertDiv.className = `alert alert-${type} d-block`;
    alertDiv.innerHTML = message;

    setTimeout(() => {
        alertDiv.className = 'alert d-none';
    }, 5000);
}

// Initial load of the table
updateTable();

// Fungsi untuk mereset hasil prediksi
function resetResult() {
    document.getElementById('result').value = '';
    document.getElementById('gender').value = '';
    document.getElementById('height').value = '';
    document.getElementById('weight').value = '';
}
