<script>
    import { FontAwesomeIcon } from '@fortawesome/svelte-fontawesome';
    import { faExternalLinkSquareAlt } from '@fortawesome/free-solid-svg-icons';
</script>

<header class=" px-[15%] border-b-[1px] border-[#2aa198] text-[#2aa198] flex items-center justify-between">
    <a href="/"><h1 class="text-[24px] text-left leading-[2.5rem]"><h1>DINO MATCH</h1></a>
    <div class="text-[20px]">
        <a href="https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes" class="px-2" target="_blank">DATASET<FontAwesomeIcon icon={faExternalLinkSquareAlt} class='ml-[6px] w-[1em] h-[1em] inline'/></a>
        <a href="https://github.com/ArmanDris/DefaultDetector" class="px-2" target="_blank">SOURCE CODE<FontAwesomeIcon icon={faExternalLinkSquareAlt} class='ml-[6px] w-[1em] h-[1em] inline'/></a>
    </div>
</header>