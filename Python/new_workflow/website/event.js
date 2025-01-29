function getQueryParam(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

// Get the 'userID' from the URL
let eventnum = getQueryParam('eventnum');

const csvData = sessionStorage.getItem('csvData');
const rows = JSON.parse(csvData);
const numevents = rows.length;

// Function to update the displayed page and buttons
function updatePage() {
    const row = rows[eventnum-1]
    console.log(row)
    document.getElementById('header_text').textContent = `Event ${eventnum} of ${numevents}`;
    document.getElementById('event_title').textContent = row['Rocket_Payload'];
    document.getElementById('event_datetime').textContent = `datetime: ${row["datetime"]}`;
    document.getElementById('event_slc').textContent = `SLC: ${row["SLC"]}`;   
    seismogram_png = 'EVENTS/' + row['datetime'].replace(' ', 'T') + '/seismic.png'
    seismogramElement = document.getElementById('seismogram');
    seismogramElement.src = seismogram_png;   

    // Disable "Previous" button if on the first page
    document.getElementById('prevBtn1').disabled = eventnum <= 1;

    // Disable "Next" button if on the last page
    document.getElementById('nextBtn1').disabled = eventnum >= numevents;

    // Disable "Previous" button if on the first page
    document.getElementById('prevBtn2').disabled = eventnum <= 1;

    // Disable "Next" button if on the last page
    document.getElementById('nextBtn2').disabled = eventnum >= numevents;

}

function previousPage() {
    if (eventnum > 1) {
        eventnum--;
        updatePage(); // Update the page and buttons
    }
}

function nextPage() {
    if (eventnum < numevents) {
        eventnum++;
        updatePage(); // Update the page and buttons
    }
}

// Initialize the page on load
updatePage();


window.onload = function() {
    const img = document.getElementById('seismogram');

    // Function to resize image based on available screen height
    function fitImageToScreen() {
        const viewportHeight = window.innerHeight; // Get the height of the viewport
        const imgAspectRatio = img.naturalWidth / img.naturalHeight; // Get the aspect ratio of the image

        // Calculate the new width while maintaining aspect ratio
        const newWidth = viewportHeight * imgAspectRatio;

        // Set the new width and height for the image
        img.style.height = `${viewportHeight}px`;
        img.style.width = `${newWidth}px`;
    }

    // Call function to fit image on page load
    fitImageToScreen();

    // Optionally, re-fit image on window resize
    window.onresize = fitImageToScreen;
};
