# RESTful API References

!!! warning

    The following contents are still under construction.

## Store RESTful API

<div id="swagger-ui"></div>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist/swagger-ui.css" />
<script src="https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js"></script>
<script>
  window.onload = () => {
    window.ui = SwaggerUIBundle({
      url: '../../assets/store-openapi.json',
      dom_id: '#swagger-ui',
      presets: [SwaggerUIBundle.presets.apis],
      layout: "BaseLayout"
    });
  }
</script>
