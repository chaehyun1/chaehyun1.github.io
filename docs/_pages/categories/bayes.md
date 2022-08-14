---
layout: archive
permalink: bayes
title: "Bayesian Stat"

author_profile: true
sidebar:
  nav: "docs"
---

{% assign posts = site.categories.bayes %}
{% for post in posts %}
  {% include custom-archive-single.html type=entries_layout %}
{% endfor %}
```
