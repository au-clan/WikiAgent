const body = document.getElementById("bodyContent")
console.log(body)

// Elements
const  initial_elements = body.children
const elements = processElements(initial_elements)
console.log("ELEMENTS")
console.log(elements)

// Coordinates
const coordinates = getCoordinates(elements)
console.log("COORDINATES")
console.log(coordinates)

// Active coordinates 
const scroll = 500
const overlap = Math.round(scroll/10)
const limitTop = coordinates[0].top
const limitBot = coordinates[coordinates.length-1].bottom
var activeTop = limitTop;
var activeBot = limitTop + scroll;

// Active elements
var [activeElements, activeElementTag, activeElementSkipTop, activeElementSkipBot] = getActiveElements()
console.log(activeElements)
console.log(activeElementSkipTop)
console.log(activeElementSkipBot)



/*
----------------------------------------------------------------------------------------------
                                        Functions
----------------------------------------------------------------------------------------------
*/

function breakTable(table){
    /*
    Breaks a table into rows
    */
    const tbody = table.getElementsByTagName("tbody")[0]
    const rows = tbody.children
    return rows
}

function processElements(elements){
    /*
    Processes the intial elements. In detail:
        - Breaks tables into rows
    */
    const processed_elements = []
    for (i=0; i<elements.length;i++){
        let currElement = elements[i]
        if (currElement.tagName == "TABLE"){
            rows = breakTable(currElement)
            processed_elements.push.apply(processed_elements, rows)
        }
        else {
            processed_elements.push(currElement)
        }
    }
    return processed_elements;
}

function getActiveElements(){
    /*
    Given all the elements and their coordinates it returns an array of the elements that are currently active, based on activeTop and activeBot
    */
    assert(elements.length == coordinates.length, "Number of elements and coordinates is not equal !")
    n = elements.length
    const activeElements = []
    const activeElementTag = []
    const activeElementSkipTop = []
    const activeElementSkipBot = []

    for (i=0; i<n; i++){
        let currElement = elements[i]
        let currCoordinates = coordinates[i]
        // Element in the active box 
        if (currCoordinates.top>=activeTop && currCoordinates.bottom<=activeBot){
            activeElementSkipTop.push(0)
            activeElementSkipBot.push(0)
            activeElementTag.push(currElement.tagName)
            activeElements.push(currElement)
        }
        // Low part of the element in the active box
        else if (currCoordinates.bottom>activeTop && currCoordinates.bottom<=activeBot){
            let overlapArea = (currCoordinates.bottom-activeTop)/currCoordinates.height;
            activeElementSkipTop.push(1-overlapArea)
            activeElementSkipBot.push(0)
            activeElementTag.push(currElement.tagName)
            activeElements.push(currElement)
        }
        // Top part of the element in the active box
        else if (currCoordinates.top>=activeTop && currCoordinates.top<activeBot){
            let overlapArea = (activeBot-currCoordinates.top)/currCoordinates.height;
            activeElementSkipTop.push(0)
            activeElementSkipBot.push(1-overlapArea)
            activeElementTag.push(currElement.tagName)
            activeElements.push(currElement)
        }
        else if(currCoordinates.top<activeTop && currCoordinates.bottom>activeBot){
            let notOverlapAreaTop = (activeTop-currCoordinates.top)/currCoordinates.height;
            let notOverlapAreaBot = (currCoordinates.bottom-activeBot)/currCoordinates.height;
            activeElementSkipTop.push(notOverlapAreaTop)
            activeElementSkipBot.push(notOverlapAreaBot)
            activeElementTag.push(currElement.tagName)
            activeElements.push(currElement)
        }
    }
    return [activeElements, activeElementTag, activeElementSkipTop, activeElementSkipBot]
}

function assert(condition, message) {
    /*
    Checks if the condition is true and throws an error if it is not
    */
    if (!condition) {
        throw new Error(message || "Assertion failed");
    }
}

function getCoordinates(arrElements){
    /*
    Given an array of elements it returns an array with the coordinates of each element
    */
    var arrCoordinates = []
    for (i=0; i<arrElements.length; i++){
        let currElement = arrElements[i]
        let currCoordinates = currElement.getBoundingClientRect()
        arrCoordinates.push(currCoordinates)
    }
    return arrCoordinates
}

function prepElements(elements){
    /* 
    Prepares elements in an array with new lines so they can be downloaded
    */
    elements_preped = []
    for (i=0; i<elements.length;i++){
        elements_preped.push(elements[i].outerHTML)
    }
    return elements_preped
}

/*
----------------------------------------------------------------------------------------------
                                        Buttons
----------------------------------------------------------------------------------------------
*/

function goDown(){
    /*
    -----------
    Down Button
    -----------
    */
    activeTop += scroll-overlap
    activeBot = activeTop + scroll
    if (activeBot>limitBot){
        activeBot = limitBot
        activeTop = limitBot-scroll
    }
    [activeElements, activeElementTag, activeElementSkipTop, activeElementSkipBot] = getActiveElements()
    console.log("New active items")
    console.log(activeElements)
    console.log(activeElementSkipTop)
    console.log(activeElementSkipBot)
}

function goUp(){
    /*
    ---------
    Up Button
    ---------
    */
    activeTop = activeTop - scroll + overlap
    activeBot = activeTop + scroll
    if (activeTop<limitTop){
        activeTop = limitTop
        activeBot = limitTop+scroll
    }
    [activeElements, activeElementTag, activeElementSkipTop, activeElementSkipBot] = getActiveElements()
    console.log("New active items")
    console.log(activeElements)
    console.log(activeElementSkipTop)
    console.log(activeElementSkipBot)
}

function downloadActiveElements(){
    const outputs = {
        'elements' : prepElements(activeElements),
        'tag' : activeElementTag,
        'skipTop' : activeElementSkipTop,
        'skipBot' : activeElementSkipBot
    }
    const out = JSON.stringify(outputs)
    var file = new Blob([out], {type: "application/json"});
    console.log("Downloading");
    var anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(file);
    anchor.download = "active_elements.json";
    anchor.click();
}


console.log("--------------------------------------------------------------------------------------------")
console.log("                                Play ground")
console.log("--------------------------------------------------------------------------------------------")
/*
----------------------------------------------------------------------------------------------
                                        Play ground                    
----------------------------------------------------------------------------------------------
*/

function test(){
    console.log("Currently just logging this :)")
}