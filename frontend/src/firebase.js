import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";       // Needed for Login
import { getFirestore } from "firebase/firestore"; // Needed for Database

// Keys from your screenshot:
const firebaseConfig = {
  apiKey: "AIzaSyAYREVFbXqO1p5J_hG1V2ozRO7toG2tiQ4",
  authDomain: "bmsit-bot-c92fd.firebaseapp.com",
  projectId: "bmsit-bot-c92fd",
  storageBucket: "bmsit-bot-c92fd.firebasestorage.app",
  messagingSenderId: "845154742634",
  appId: "1:845154742634:web:a42fa53daaa780b237f513",
  measurementId: "G-B9R4KKDJJQ"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Export these so the rest of the app can use them
export const auth = getAuth(app);
export const db = getFirestore(app);