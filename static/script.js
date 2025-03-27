function uploadImage() {
    let input = document.getElementById("imageUpload");
    if (input.files.length === 0) {
        alert("Please select an image first!");
        return;
    }

    let formData = new FormData();
    formData.append("image", input.files[0]);

    fetch("/detect", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        let resultDiv = document.getElementById("result");
        if (data.error) {
            resultDiv.innerHTML = `<p style="color: red;">${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `<p><strong>Gender:</strong> ${data.gender}</p>
                                   <p><strong>Age:</strong> ${data.age}</p>`;
        }
    })
    .catch(error => console.error("Error:", error));
}
