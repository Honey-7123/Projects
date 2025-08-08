// Register
document.getElementById("registerForm")?.addEventListener("submit", function (e) {
  e.preventDefault();
  const name = document.getElementById("name").value;
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  localStorage.setItem("user", JSON.stringify({ name, email, password }));
  alert("Registration successful! Please login.");
  window.location.href = "login.html";
});

// Login
document.getElementById("loginForm")?.addEventListener("submit", function (e) {
  e.preventDefault();
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  const user = JSON.parse(localStorage.getItem("user"));
  if (user && user.email === email && user.password === password) {
    localStorage.setItem("loggedIn", "true");
    localStorage.setItem("loggedUser", user.name);
    window.location.href = "index.html";
  } else {
    alert("Invalid credentials");
  }
});

// Logout
function logout() {
  localStorage.removeItem("loggedIn");
  localStorage.removeItem("loggedUser");
  window.location.href = "login.html";
}

// Show welcome
window.onload = function () {
  const user = localStorage.getItem("loggedUser");
  if (user && document.getElementById("welcome")) {
    document.getElementById("welcome").textContent = `Welcome, ${user}`;
  }
};