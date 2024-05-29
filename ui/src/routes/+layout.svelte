<script>
    import '../global.css';
    import Header from "$lib/components/Header.svelte";
    import Footer from "$lib/components/Footer.svelte";
</script>

<svelte:head>
  <title>Dino Match</title>
  <meta name="description" content="Finds similar images using Dino embedder and QDrant!">
</svelte:head>

<Header />
<slot />
<Footer />