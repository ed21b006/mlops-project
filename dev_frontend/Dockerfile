FROM nginx:1.25-alpine
RUN apk add --no-cache apache2-utils
RUN rm -rf /usr/share/nginx/html/*
RUN htpasswd -bc /etc/nginx/.htpasswd admin admin
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html predict.html styles.css app.js /usr/share/nginx/html/
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
