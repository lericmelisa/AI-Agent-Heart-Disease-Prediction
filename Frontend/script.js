function clearForm() {
    // Clear form inputs
    $("#age").val(""); // Clear age input
    $("#cp").val(""); // Clear cp input
    $("#chol").val(""); // Clear chol input
    $("#thalach").val(""); // Clear thalach input
    $("#oldpeak").val(""); // Clear oldpeak input
}


function getFeatureImportances() {
    fetch('http://127.0.0.1:5000/feature_importances', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        const chartDiv = document.getElementById('featureImportanceChart');
        if (!chartDiv) {
            console.error('Element with ID "featureImportanceChart" not found.');
            return;
        }

        chartDiv.style.display = 'block'; // Make the chart visible
        chartDiv.innerHTML = ''; // Clear any existing chart content

        // Render the Plotly chart using the data from the API
        Plotly.newPlot(chartDiv, data.data, data.layout);
    })
    .catch(error => {
        console.error('Error fetching feature importances:', error);
        alert('Failed to fetch feature importances. Check console for details.');
    });
}






function retrainModel() {
    fetch('http://127.0.0.1:5000/retrain_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => response.json())
    .then(result => {
        // Display success message
        document.getElementById('result').style.display = 'block';
        document.getElementById('result').innerHTML = `
            <p><strong>Message:</strong> ${result.message}</p>
        `;
    })
    .catch(error => {
        // Display error message
        document.getElementById('result').style.display = 'block';
        document.getElementById('result').innerHTML = `<p style="color: red;">Error: Unable to retrain the model.</p>`;
        console.error('Error:', error);
    });
}



document.addEventListener('DOMContentLoaded', function() {

    // Predaja forme
    document.getElementById("predictionForm").addEventListener("submit", async function (e) {
        e.preventDefault();

        // Prikupljanje podataka iz forme
        const data = {
            age: document.getElementById('age').value,
            cp: document.getElementById('cp').value,
            chol: document.getElementById('chol').value,
            thalach: document.getElementById('thalach').value,
            oldpeak: document.getElementById('oldpeak').value
            
        };
        clearForm();  
    console.log("Forma je ociscena");

        // Log the data to check it's correct
        console.log("Data being sent to server:", data);

        // Slanje POST zahtjeva na server
        try {
            const response = await fetch('http://127.0.0.1:5000/predvidi', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            // Check if the response is successful
            const result = await response.json();
            console.log("Received response:", result);

            const resultDiv = document.getElementById('result');

            if (result.error) {
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = `<p style="color: red;">Error: ${result.error}</p>`;
            } else {
                resultDiv.style.display = "block";
                resultDiv.innerHTML = `
                    <p><strong>Predikcija:</strong> ${result.Predikcija}</p>
                    <p><strong>Pouzdanost:</strong></p>
                    <ul>
                        <li><strong>Ima srčanu bolest:</strong> ${result.Pouzdanost['Ima srčanu bolest']}</li>
                        <li><strong>Nema srčanu bolest:</strong> ${result.Pouzdanost['Nema srčanu bolest']}</li>
                    </ul>
                `;
               
                  

            }
        } catch (error) {
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = `<p style="color: red;">Error: Unable to process the request. ${error.message}</p>`;
            console.error('Error:', error);
        }
     
     
    });

    // Dugme za učitavanje grafa
    const loadPlotButton = document.getElementById("load-plot");
    const plotDiv = document.getElementById("plot");

    // URL API endpoint-a za graf
    const plotApiUrl = "http://127.0.0.1:5000/visualize";

    loadPlotButton.addEventListener("click", function () {
        loadPlotButton.textContent = "Učitavam graf...";
        loadPlotButton.disabled = true;

        // Poziv API-ju za dobijanje grafa
        fetch(plotApiUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                const graphData = JSON.parse(data.graph);
                Plotly.newPlot("plot", graphData.data, graphData.layout);
                plotDiv.style.display = "block";
                loadPlotButton.textContent = "Prikazi graf";
                loadPlotButton.disabled = false;
            })
            .catch(error => {
                console.error("Error loading plot:", error);
                alert("Greška prilikom učitavanja grafa. Pogledajte konzolu za detalje.");
                loadPlotButton.textContent = "Prikazi Graf";
                loadPlotButton.disabled = false;
            });
    });

    


}); 



