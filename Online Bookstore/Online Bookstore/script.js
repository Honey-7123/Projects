const books = [
  {
    id: 1,
    title: "The Alchemist",
    author: "Paulo Coelho",
    price: 299,
    category: "Fiction",
    img: "https://m.media-amazon.com/images/I/51Z0nLAfLmL.jpg"
  },
  {
    id: 3,
    title: "Ikigai",
    author: "Francesc Miralles",
    price: 250,
    category: "Non-Fiction",
    img: "https://m.media-amazon.com/images/I/81l3rZK4lnL.jpg"
  },
  {
    id: 4,
    title: "Harry Potter",
    author: "J.K. Rowling",
    price: 650,
    category: "Fiction",
    img: "https://m.media-amazon.com/images/I/81YOuOGFCJL.jpg"
  }
];

let cart = [];

function displayBooks(bookList) {
  const container = document.getElementById("book-list");
  container.innerHTML = "";
  bookList.forEach(book => {
    const bookDiv = document.createElement("div");
    bookDiv.className = "book";
    bookDiv.innerHTML = `
      <img src="${book.img}" alt="${book.title}" />
      <h3>${book.title}</h3>
      <p><em>${book.author}</em></p>
      <p>₹${book.price}</p>
      <button onclick="addToCart(${book.id})">Add to Cart</button>
    `;
    container.appendChild(bookDiv);
  });
}

function addToCart(id) {
  const book = books.find(b => b.id === id);
  cart.push(book);
  updateCart();
}

function updateCart() {
  const cartList = document.getElementById("cart-items");
  cartList.innerHTML = "";
  let total = 0;
  cart.forEach((book) => {
    total += book.price;
    const li = document.createElement("li");
    li.textContent = `${book.title} - ₹${book.price}`;
    cartList.appendChild(li);
  });
  document.getElementById("total").textContent = total;
}

function filterBooks(category) {
  if (category === "all") {
    displayBooks(books);
  } else {
    const filtered = books.filter(book => book.category === category);
    displayBooks(filtered);
  }
}

function simulatePayment() {
  if (cart.length === 0) {
    alert("Cart is empty!");
  } else {
    alert("Payment Successful! Thank you for your purchase.");
    cart = [];
    updateCart();
  }
}

window.onload = () => {
  if (document.getElementById("book-list")) {
    displayBooks(books);
  }
};