<script>
    import { onMount } from "svelte";
    import { faForward, faArrowsRotate } from '@fortawesome/free-solid-svg-icons';
    import { FontAwesomeIcon } from '@fortawesome/svelte-fontawesome';

    let images = {}

    let similar_images = {}

    let selected_image = "agri_0_9266.jpeg"

    async function getSampleImages() {
        fetch("http://localhost:5000/getSample")
            .then((response) => response.json())
            .then((data) => {
                images = {...data};
                selected_image = Object.keys(images)[0];
            })
            .catch((error) => console.error("Error fetching images:", error));
    }

    async function runPrediction() {
        fetch("http://localhost:5000/getSimilar", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({'file_name': selected_image})
        })
            .then((response) => response.json())
            .then((data) => {
                similar_images = {...data};
            })
            .catch((error) => console.error("Error fetching images:", error))
    }

    onMount(() => {
        getSampleImages();
    });
</script>

<main class="w-full h-full">
    <h1 class="text-black font-serif text-[44px]">Sample Images:</h1>
    <div class="flex items-center justify-between">
        {#if Object.keys(images).length > 0}
            {#each Object.entries(images) as [key, value]}
                <img on:click={() => selected_image=key} src={"data:image/jpeg;base64," + value} alt={key}/>
            {/each}
        {/if}
    </div>
    <button class="bg-[#d3368240] text-[#d33682]" on:click={getSampleImages}>Get New Sample<FontAwesomeIcon icon={faArrowsRotate} class="w-[1em] h-[1em] px-2 inline" /></button>
    <button class="bg-[#85990040] text-[#859900]" on:click={runPrediction}>Find Similar Images<FontAwesomeIcon icon={faForward} class="w-[1em] h-[1em] px-2 inline" /></button>

    <div class="flex justify-between flex-wrap p-6">
        {#if Object.keys(similar_images).length > 0}
            {#each Object.entries(similar_images) as [key, value]}
                <div>
                    <p>Similarity: {value[0]}</p>
                    <img src={"data:image/jpeg;base64," + value[1]} alt={key}>
                </div>
            {/each}
        {/if}
    </div>
</main>

<style>
    img {
        width: 224px;
        height: 224px;
    }

    button {
        font-family: sans-serif;
        font-weight: bold;
        padding: 14px;
        margin-right: 14px;
        margin-top: 14px;
        border-radius: 14px;
    }
</style>