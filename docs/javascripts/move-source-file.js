// Copyright (c) Microsoft. All rights reserved.

document.addEventListener('DOMContentLoaded', function () {
  const container = document.querySelector('.md-content__inner');
  if (!container) return;
  const firstH1 = container.querySelector('h1');
  const meta = container.querySelector('.md-source-file'); // the block rendered by source-file.html
  if (firstH1 && meta && meta.previousElementSibling !== firstH1) {
    firstH1.insertAdjacentElement('afterend', meta);
  }
});
