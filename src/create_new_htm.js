const fs = require('fs');
const path = require('path');
const yargs = require('yargs');
var { Readability } = require('@mozilla/readability');
var { JSDOM } = require('jsdom');

// Define the command line arguments using yargs.
const argv = yargs
  .option('sourceFile', {
    alias: 's',
    describe: 'Path to the source HTML file',
    demandOption: true, // Makes the source file path a required argument
    type: 'string',
  })
  .option('destinationFolder', {
    alias: 'd',
    describe: 'Path to the destination folder',
    demandOption: true, // Makes the destination folder path a required argument
    type: 'string',
  })
  .argv;

// Get the source file path from the command line arguments.
const sourceFilePath = path.resolve(argv.sourceFile);
const destinationFilePath = path.resolve(argv.destinationFolder);


// Read the HTML file and save it, as shown in the previous example.
fs.readFile(sourceFilePath, 'utf8', (err, data) => {
  if (err) {
    console.error('Error reading the file:', err);
    return;
  }
    var doc = new JSDOM(data.toString());
    let reader = new Readability(doc.window.document);
    let article = reader.parse();
  // Call a function to save the HTML content.
  saveHTML(article.content);
});

function saveHTML(htmlContent) {
  fs.writeFile(destinationFilePath, htmlContent, 'utf8', (err) => {
    if (err) {
      console.error('Error saving the file:', err);
    } else {
      console.log('HTML file saved successfully.');
    }
  });
}
