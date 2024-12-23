$(document).ready(function () {
    // Fullscreen toggle functionality
    $('[data-widget="fullscreen"]').on('click', function () {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        }
    });

    // Show the form and split view when the "Prediction with AI" button is clicked
    $("#btn-ai").click(function () {
        $(".container-fluid").addClass("split-view"); // Add split-view class
        $("#ai-form").fadeIn(); // Show form with fade effect
    });

    // Hide the form and remove split view when the close button is clicked
    $("#btn-close-form").click(function () {
        $("#ai-form").fadeOut(function () {
            $(".container-fluid").removeClass("split-view"); // Remove split-view class after form is hidden
        });
    });

    // Handle the prediction logic
    function calculatePrediction(gender, height, weight) {
        if (!height || !weight) {
            alert("Please fill in height and weight!");
            return null;
        }

        const bmi = weight / ((height / 100) ** 2);
        let status = "";

        // Determine BMI category
        if (bmi < 16) {
            status = "Severely Underweight";
        } else if (bmi >= 16 && bmi < 18.5) {
            status = "Underweight";
        } else if (bmi >= 18.5 && bmi < 25) {
            status = "Normal weight";
        } else if (bmi >= 25 && bmi < 30) {
            status = "Overweight";
        } else if (bmi >= 30 && bmi < 35) {
            status = "Obese Class I";
        } else if (bmi >= 35 && bmi < 40) {
            status = "Obese Class II";
        } else {
            status = "Obese Class III";
        }

        return { bmi, status };
    }

    // Handle the "Predict Status" button click
    $("#btn-predict").click(function () {
        const gender = $("#gender").val();
        const height = parseFloat($("#height").val());
        const weight = parseFloat($("#weight").val());

        const result = calculatePrediction(gender, height, weight);
        if (result) {
            const { bmi, status } = result;
            // Display the prediction result
            $("#prediction").val(`${status} (BMI: ${bmi.toFixed(2)})`);

            // Add a new record dynamically after prediction
            const newRow = `
                <tr>
                    <td>${$("#record-table tr").length + 1}</td>
                    <td>${gender}</td>
                    <td>${height}</td>
                    <td>${weight}</td>
                    <td>${bmi.toFixed(2)}</td>
                    <td>${status}</td>
                    <td>
                        <button class="btn btn-danger btn-sm delete-record">
                            <i class="fas fa-trash-alt"></i>
                        </button>
                    </td>
                    <td>
                        <button class="btn btn-warning btn-sm edit-record">
                            <i class="fas fa-edit"></i>
                        </button>
                    </td>
                </tr>
            `;
            $("#record-table").append(newRow);

            // Optionally reset the form
            $("#gender").val("");
            $("#height").val("");
            $("#weight").val("");
            $("#prediction").val("");
        }
    });

    // Delete record functionality
    $(document).on("click", ".delete-record", function () {
        if (confirm("Are you sure you want to delete this record?")) {
            $(this).closest("tr").remove();
        }
    });

    // Edit record functionality
    $(document).on("click", ".edit-record", function () {
        const row = $(this).closest("tr");
        const gender = row.find("td:eq(1)").text();
        const height = row.find("td:eq(2)").text();
        const weight = row.find("td:eq(3)").text();

        // Populate the form with the data
        $("#gender").val(gender);
        $("#height").val(height);
        $("#weight").val(weight);

        // Show the form and activate split view
        $(".container-fluid").addClass("split-view");
        $("#ai-form").fadeIn();
    });

    // Implement dragging of the form
    let isDragging = false;
    let offsetX, offsetY;

    // When the user starts dragging
    $("#form-header").on("mousedown", function (e) {
        isDragging = true;
        offsetX = e.clientX - $("#ai-form").offset().left;
        offsetY = e.clientY - $("#ai-form").offset().top;
        $(document).on("mousemove", moveForm);
        $(document).on("mouseup", stopDragging);
    });

    // Function to move the form
    function moveForm(e) {
        if (isDragging) {
            const left = e.clientX - offsetX;
            const top = e.clientY - offsetY;
            $("#ai-form").css({ left: left, top: top });
        }
    }

    // Stop dragging when the mouse is released
    function stopDragging() {
        isDragging = false;
        $(document).off("mousemove", moveForm);
        $(document).off("mouseup", stopDragging);
    }
});
