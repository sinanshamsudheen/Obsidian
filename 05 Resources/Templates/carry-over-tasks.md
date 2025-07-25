<%*
const moment = tp.date.now("YYYY-MM-DD", 0); 
const yesterday = tp.date.now("YYYY-MM-DD", -1);

const vault = app.vault;
const dailyFolder = "08 Work"; // Change this to your daily notes folder
const yesterdayFilePath = `${dailyFolder}/${yesterday}.md`;

let tasksToCarryOver = [];

try {
  // Try to read yesterday's file
  const file = vault.getAbstractFileByPath(yesterdayFilePath);
  if (file && file.extension === "md") {
    const content = await vault.cachedRead(file);

    // Extract all unchecked tasks using regex
    const regex = /^- \[ \] .+$/gm;
    let matches = [...content.matchAll(regex)];

    if (matches.length > 0) {
      tasksToCarryOver = matches.map(m => m[0]);
    }
  }
} catch (e) {
  tasksToCarryOver = [];
}

if (tasksToCarryOver.length > 0) { %>
## Tasks Carried Over from <%+ yesterday %>

<% tasksToCarryOver.forEach(task => { %>
<%- task %>
<% }) %>

<%* } else { %>
## Tasks Carried Over from <%+ yesterday %>

_No tasks to carry over from yesterday._
<%* } %>
