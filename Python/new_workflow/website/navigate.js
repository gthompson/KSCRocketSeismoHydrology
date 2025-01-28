// Array of URLs (Replace these URLs with actual web page URLs you want to navigate through)
const pages = [
    'launch1.html',
    'launch2.html',
    'launch3.html'
];

/*
const fs = require('fs');
const path = require('path');

// Directory to scan for HTML files
const directoryPath = './launches'; // Replace with the path to your directory containing web pages

// Function to read all HTML files in a directory
function generateWebpageList(directoryPath) {
    fs.readdir(directoryPath, (err, files) => {
        if (err) {
            console.error("Error reading directory:", err);
            return;
        }

        // Filter out only .html files
        const htmlFiles = files.filter(file => path.extname(file).toLowerCase() === '.html');

        // Map the file names to create full URL paths (optional: modify the base URL as needed)
        const webpageUrls = htmlFiles.map(file => path.join(directoryPath, file));

        // Output the array of webpage URLs
        console.log(htmlFiles)
    });

}   

// Call the function to generate the webpage list
generateWebpageList(directoryPath);
*/

let currentIndex = 0; // Start at the first page



// Function to update the displayed page and buttons
function updatePage() {
    const currentPage = pages[currentIndex];
    let launchnum = currentIndex + 1
    document.getElementById('header_text').textContent = `Launch ${launchnum} of ${pages.length}`;
    //document.getElementById('launchnum').textContent = `Launch ${launchnum} of ${pages.length}`;
    //document.getElementById('launch-page').textContent = `Current Page: ${currentPage}: Launch ${launchnum} of ${pages.length}`;
    document.getElementById('launch_title').textContent = `Current Page: ${currentPage}: Launch ${launchnum} of ${pages.length}`;
    document.getElementById('launch_date').textContent = `September 1st, ${currentIndex + 2022}`;
    document.getElementById('launch_description').textContent = "The Falcon 9 rocket successfully launched from Kennedy Space Center carrying a communications satellite";
    document.getElementById('launch_details').textContent = "More details about the Falcon 9 launch...";

    //document.getElementById('current-page').textContent = `Current Page: ${currentPage}`;

    // Disable "Previous" button if on the first page
    document.getElementById('prevBtn').disabled = currentIndex === 0;

    // Disable "Next" button if on the last page
    document.getElementById('nextBtn').disabled = currentIndex === pages.length - 1;
}

function previousPage() {
    if (currentIndex > 0) {
        currentIndex--;
        updatePage(pages, currentIndex); // Update the page and buttons
    }
    //console.log(currentIndex);

}

function nextPage() {
    if (currentIndex < pages.length - 1) {
        currentIndex++;
        updatePage(pages, currentIndex); // Update the page and buttons
    }
    //console.log(currentIndex);
}

// Event listener for the "Previous" button
const prevElement = document.getElementById('prevBtn')
prevElement.addEventListener('click', previousPage())

// Event listener for the "Next" button
const nextElement = document.getElementById('nextBtn')
nextElement.addEventListener('click', nextPage())

// Initialize the page on load
updatePage(pages, currentIndex);

// Function to fetch and parse a CSV file
function fetchCSV() {
    const csvFileUrl = 'http://131.247.211.6/usfseismiclab.org/html/rocketcat/launches.csv';  // URL to your CSV file (replace with correct path)

    fetch(csvFileUrl)
        .then(response => response.text())  // Get the CSV content as text
        .then(csvData => {
            const parsedData = parseCSV(csvData);
            const jsonText = JSON.stringify(parsedData, null, 2);
            const jsonData = JSON.parse(jsonText);
            //document.getElementById('output').textContent = jsonText;
            document.getElementById("jsonTable").appendChild(generateTable(jsonData));
            //console.log(parsedData);
        })
        .catch(error => {
            console.error('Error fetching CSV:', error);
            document.getElementById('jsonTable').textContent = 'Error loading CSV file.';
        });
}

// Function to parse CSV text into an array of objects
function parseCSV(csvText) {
    const rows = csvText.split('\n');  // Split the CSV text into rows
    const headers = rows[0].split(',');  // The first row contains headers

    return rows.slice(1).map(row => {
        const values = row.split(',');
        const obj = {};
        console.log(values)
        if (values.length > 0) {
            headers.forEach((header, index) => {
                obj[header.trim()] = values[index].trim();
            });
        }

        return obj;
    });
}

// Call the function to fetch and parse the CSV file
fetchCSV();

// Parse the JSON text into a JavaScript object (array of objects)
//const jsonData = JSON.parse(jsonText);

// Function to generate an HTML table from JSON data
function generateTable(data) {
    // Create a table element
    const table = document.createElement("table");
    table.style.border = "1px solid black";
    table.style.borderCollapse = "collapse";
    table.style.width = "100%";

    // Create the table header row
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    // Get the keys from the first object (use them as table headers)
    const headers = Object.keys(data[0]);
    headers.forEach(header => {
        const th = document.createElement("th");
        th.style.border = "1px solid black";
        th.style.padding = "8px";
        th.textContent = header;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create the table body rows
    const tbody = document.createElement("tbody");
    data.forEach(rowData => {
        const row = document.createElement("tr");
        headers.forEach(header => {
            const td = document.createElement("td");
            td.style.border = "1px solid black";
            td.style.padding = "8px";
            td.textContent = rowData[header];
            row.appendChild(td);
        });
        tbody.appendChild(row);
    });
    table.appendChild(tbody);

    return table;
}

// Append the generated table to the document
//document.getElementById("jsonTable").appendChild(generateTable(jsonData));
