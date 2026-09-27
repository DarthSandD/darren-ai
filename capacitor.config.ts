import { defineConfig } from "@capacitor/cli";

/** Capacitor config — wraps the same /web UI as a native Android app.
 *  Point `server.url` at your deployed backend origin, OR bundle web/ locally
 *  and set window.DARREN_API in the app to the API host. */
export default defineConfig({
  appId: "com.darrenai.app",
  appName: "Darren Ai",
  webDir: "web",
  server: {
    androidScheme: "https",
    // Uncomment to load the live site instead of the bundled files:
    // url: "https://your-domain.example",
    // cleartext: false,
  },
  android: {
    allowMixedContent: false,
  },
});
