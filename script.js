function handleSubmit(event) {
  event.preventDefault() // Prevent form submission to keep the page from reloading

  // Get form data
  const grievanceText = document.getElementById("grievanceText").value
  const locationText = document.getElementById("locationText").value
  const grievanceUpload = document.getElementById("grievanceUpload").files[0]

  // Create a new URL for the new page (or you can open a new window)
  const newWindow = window.open("", "_blank") // Open a blank window

  // Create HTML content to display the grievance details
  let content = `
    <h1>Grievance Submitted</h1>
    <p><strong>Grievance:</strong> ${grievanceText}</p>
    <p><strong>Location:</strong> ${locationText}</p>
  `

  // If a file was uploaded, add it to the content
  if (grievanceUpload) {
    content += `<p><strong>Supporting Document:</strong> ${grievanceUpload.name}</p>`

    // If the uploaded file is an image, display it
    if (grievanceUpload.type.startsWith("image/")) {
      const reader = new FileReader()
      reader.onload = function (e) {
        const imageSrc = e.target.result
        content += `<p><strong>Uploaded Image:</strong></p><img src="${imageSrc}" alt="Uploaded Image" style="max-width: 500px; max-height: 500px;"/><br>`
        // Write the content to the new window
        newWindow.document.write(content)
        newWindow.document.close()
      }
      reader.readAsDataURL(grievanceUpload) // Read the image as a data URL
      return // Exit the function while the image is being loaded
    }
  }

  // Write the content to the new window (for non-image files or after image is loaded)
  newWindow.document.write(content)
  newWindow.document.close()
}
